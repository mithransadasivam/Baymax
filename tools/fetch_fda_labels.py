"""Download FDA-approved drug label text (via openFDA, from the NLM's DailyMed SPL data) for the
medicines people most often ask about, into the knowledge library.

Drug labels are the FDA-reviewed source of truth for dosing, contraindications, warnings, and
interactions. Text is kept verbatim (whitespace collapsed). Each entry records the label's set id,
version, effective date, and manufacturer, so a statement can be traced to one exact label.

    uv run python tools/fetch_fda_labels.py [--only metformin,ibuprofen]
"""

from __future__ import annotations

import json
import re
import sys
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import sourcelib as sl  # noqa: E402

API = "https://api.fda.gov/drug/label.json"
SOURCE = sl.ALLOWED_HOSTS["api.fda.gov"]

# Generic names. Common prescription drugs by US prescription volume and class, plus the OTC
# medicines people ask about most.
DRUGS = """
acetaminophen ibuprofen naproxen aspirin diphenhydramine loratadine cetirizine fexofenadine
famotidine omeprazole esomeprazole lansoprazole loperamide guaifenesin dextromethorphan
pseudoephedrine phenylephrine simethicone bismuth docusate bisacodyl senna meclizine
chlorpheniramine hydroxyzine ranitidine calcium sucralfate ondansetron promethazine
metformin glipizide glimepiride glyburide pioglitazone sitagliptin linagliptin empagliflozin
dapagliflozin canagliflozin semaglutide liraglutide dulaglutide insulin levothyroxine
liothyronine methimazole propylthiouracil
atorvastatin simvastatin rosuvastatin pravastatin lovastatin ezetimibe fenofibrate gemfibrozil
lisinopril enalapril ramipril benazepril losartan valsartan irbesartan olmesartan telmisartan
amlodipine nifedipine diltiazem verapamil metoprolol atenolol carvedilol propranolol bisoprolol
nebivolol hydrochlorothiazide chlorthalidone furosemide torsemide bumetanide spironolactone
clonidine hydralazine isosorbide nitroglycerin ranolazine digoxin amiodarone
warfarin apixaban rivaroxaban dabigatran clopidogrel ticagrelor prasugrel
sertraline fluoxetine citalopram escitalopram paroxetine venlafaxine duloxetine bupropion
mirtazapine trazodone amitriptyline nortriptyline buspirone lithium lamotrigine
aripiprazole quetiapine risperidone olanzapine haloperidol
alprazolam lorazepam clonazepam diazepam zolpidem eszopiclone temazepam
methylphenidate amphetamine atomoxetine
gabapentin pregabalin topiramate levetiracetam carbamazepine phenytoin valproate
sumatriptan rizatriptan
tramadol oxycodone hydrocodone morphine codeine fentanyl buprenorphine naloxone naltrexone
methadone cyclobenzaprine methocarbamol tizanidine baclofen carisoprodol
prednisone prednisolone methylprednisolone dexamethasone hydrocortisone fluticasone
budesonide mometasone triamcinolone
albuterol salbutamol ipratropium tiotropium montelukast theophylline
amoxicillin penicillin cephalexin cefuroxime cefdinir azithromycin clarithromycin
doxycycline minocycline ciprofloxacin levofloxacin moxifloxacin trimethoprim
sulfamethoxazole nitrofurantoin metronidazole clindamycin vancomycin linezolid
fluconazole terbinafine nystatin acyclovir valacyclovir famciclovir oseltamivir
ivermectin hydroxychloroquine albendazole
sildenafil tadalafil finasteride tamsulosin oxybutynin tolterodine mirabegron
estradiol medroxyprogesterone norethindrone levonorgestrel progesterone
alendronate raloxifene allopurinol febuxostat colchicine
methotrexate azathioprine adalimumab
sumatriptan ondansetron metoclopramide pantoprazole
folic cyanocobalamin cholecalciferol ergocalciferol ferrous potassium magnesium
epinephrine
""".split()


def pick(drug: str) -> dict | None:
    """The newest single-ingredient label for ``drug`` that has the sections worth keeping."""
    query = f'openfda.generic_name:"{drug}"'
    url = f"{API}?search={urllib.parse.quote(query, safe=':\"')}&limit=25"
    try:
        data = json.loads(sl.fetch(url, pause=0.3))
    except Exception as exc:  # noqa: BLE001
        print(f"  {drug}: lookup failed ({exc})")
        return None
    best = None
    for record in data.get("results", []):
        names = [n.lower() for n in record.get("openfda", {}).get("generic_name", [])]
        if len(names) != 1 or drug.lower() not in names[0] or re.search(r" and |,|/| with |\+", names[0]):
            continue  # combination products and look-alike names are skipped
        if not (record.get("indications_and_usage") and (record.get("dosage_and_administration") or record.get("warnings"))):
            continue
        # Prefer completeness first, then recency.
        score = (sum(1 for k in SECTIONS_WANTED if record.get(k)), record.get("effective_time", ""))
        if best is None or score > best[0]:
            best = (score, record)
    return best[1] if best else None


# (field(s), heading, max passages). OTC labels split warnings across several fields.
SECTIONS = [
    (("boxed_warning",), "Boxed warning", 2),
    (("indications_and_usage", "purpose"), "Uses", 2),
    (("dosage_and_administration",), "Dosage and administration", 4),
    (("contraindications",), "Contraindications", 2),
    (("drug_interactions",), "Drug interactions", 10),
    (("warnings_and_cautions", "warnings", "do_not_use", "ask_doctor", "ask_doctor_or_pharmacist", "when_using", "stop_use"), "Warnings and precautions", 3),
    (("adverse_reactions",), "Adverse reactions", 2),
    (("use_in_specific_populations", "pregnancy", "pregnancy_or_breast_feeding"), "Pregnancy and specific populations", 2),
    (("overdosage",), "Overdose", 2),
]
SECTIONS_WANTED = [f for fields, _, _ in SECTIONS for f in fields]


def entries_for(drug: str, record: dict) -> list[str]:
    openfda = record.get("openfda", {})
    name = (openfda.get("generic_name") or [drug])[0].title()
    brands = [b.title() for b in openfda.get("brand_name", [])[:4]]
    classes = openfda.get("pharm_class_epc", []) + openfda.get("pharm_class_moa", [])
    maker = (openfda.get("manufacturer_name") or ["unknown manufacturer"])[0]
    effective = record.get("effective_time", "")
    effective = f"{effective[:4]}-{effective[4:6]}-{effective[6:8]}" if len(effective) == 8 else effective
    set_id = record.get("set_id", "")
    url = f"https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid={set_id}"
    source = f"{SOURCE}, {name} ({maker}), effective {effective}"
    out: list[str] = []
    for fields, heading, limit in SECTIONS:
        pieces: list[str] = []
        for field in fields:
            for text in record.get(field, []):
                cleaned = sl.clean_label_text(text)
                if len(cleaned) > 40:
                    pieces.append(cleaned)
        if not pieces:
            continue
        parts = sl.passages("\n".join(pieces))[:limit]
        for number, piece in enumerate(parts, start=1):
            suffix = f" (part {number} of {len(parts)})" if len(parts) > 1 else ""
            keywords = [name, drug, *brands, heading, *classes, *drug.split()]
            out.append(
                sl.entry(f"{name}: {heading}{suffix}", source, url, sl.today(), keywords, piece)
                .replace("URL: ", f"Label-set-id: {set_id}\nURL: ", 1)
            )
    return out


def main() -> None:
    only = None
    if "--only" in sys.argv:
        only = {d.strip().lower() for d in sys.argv[sys.argv.index("--only") + 1].split(",")}
    drugs = [d for d in dict.fromkeys(DRUGS) if not only or d in only]
    all_entries: list[str] = []
    found: dict[str, dict] = {}
    missing: list[str] = []
    for index, drug in enumerate(drugs, start=1):
        record = pick(drug)
        if record is None:
            missing.append(drug)
            print(f"[{index}/{len(drugs)}] {drug}: no suitable label")
            continue
        entries = entries_for(drug, record)
        all_entries.extend(entries)
        found[drug] = {
            "set_id": record.get("set_id"),
            "version": record.get("version"),
            "effective_time": record.get("effective_time"),
            "manufacturer": (record.get("openfda", {}).get("manufacturer_name") or [""])[0],
            "passages": len(entries),
        }
        print(f"[{index}/{len(drugs)}] {drug}: {len(entries)} passages")
    header = (
        "FDA-approved drug labels via openFDA / DailyMed (National Library of Medicine). Public domain. "
        "Text is verbatim from the label apart from whitespace. Each entry names its label set id."
    )
    path = sl.write_topic_file("fda_labels.md", header, all_entries)
    sl.update_manifest(
        "fda_labels",
        {
            "authority": "US Food and Drug Administration (labels distributed by NLM DailyMed)",
            "api": API,
            "retrieved": sl.today(),
            "drugs": found,
            "no_suitable_label": missing,
            "passages": len(all_entries),
        },
    )
    print(f"{len(found)} drugs, {len(all_entries)} passages -> {path}; missing: {missing}")


if __name__ == "__main__":
    main()
