# Heart rate from your Apple Watch

Baymax can read a heart-rate reading from your Apple Watch, keep it on your PC, and explain it when you ask. Nothing goes to the cloud: a Shortcut on your iPhone sends the number straight to Baymax over your home WiFi or over Tailscale.

> **A wellness reading, not a medical measurement.** A consumer watch can be wrong (a loose strap, a cold wrist, movement). Baymax explains the number in plain language and says when to see a doctor, but it won't diagnose a rhythm problem. If you have chest pain, trouble breathing, or you feel faint, call 911.

## What you need

- Baymax running with the phone server on (the desktop shortcut does this: `--phone`).
- Tailscale on your PC and your iPhone, both signed in, if you want it to work away from home. At home on the same WiFi it also works without Tailscale.
- Your **token**, which stops anyone else on the network from adding fake readings. It's created on first launch in this file on your PC:

  `C:\Users\<you>\.baymax\vitals-token`

  Open it in Notepad and copy the text.

## Build the Shortcut (about 5 minutes, once)

On your iPhone, open **Shortcuts** → **+** and name it **Baymax Heart**. Add these actions in order:

1. **Find Health Samples**. Type **Heart Rate**, Sort by **Start Date**, **Latest First**, Limit **1**. (The first run asks permission to read Heart Rate. Allow it.)
2. **Get Contents of URL**
   - URL: `https://<your-pc-name>.<your-tailnet>.ts.net:8544/vitals` (Baymax prints the exact address as "Tailscale: open …" when it starts). On the same WiFi you can use `https://<pc-ip>:8543/vitals`, after you've trusted the certificate once.
   - Method: **POST**
   - Request Body: **JSON**, with these fields:
     - `token` (Text): paste your token
     - `heart_rate` (Number): choose the **Value** of the Health Sample from step 1
     - `time` (Text): the sample's **Start Date**, formatted as ISO 8601
     - `explain` (Text): `true`
3. **Speak Text** (or **Show Result**) with the **Contents of URL**.

When you run it, Baymax stores the reading and replies with a calm explanation, which your phone or watch reads aloud.

### Optional extras
- Add a second **Find Health Samples** for **Resting Heart Rate** and send it as `resting_heart_rate`. Baymax then keeps a short history of your resting values.
- Send only the number, with no explanation: leave out `explain` and Baymax just replies "Got it. 72 beats per minute."

## Use it on the Watch and by voice

- In the Shortcut's settings (the **i** button), turn on **Show on Apple Watch**. It then appears in the Shortcuts app on the watch.
- Say **"Hey Siri, Baymax Heart"**, or add it as a watch-face complication.
- Then, at the PC or on the phone page, you can ask Baymax in plain words: *"What's my heart rate?"*, *"Is my resting heart rate okay?"*, *"My heart feels like it's racing."* It uses the latest reading it received, and it tells you if that reading is old.

## How Baymax uses the reading

| Situation | What happens |
|---|---|
| You ask about your heart rate or pulse | The latest reading, its age, and the normal range for adults at rest are handed to the model. It explains calmly, leading with everyday causes (caffeine, stress, a flight of stairs, poor sleep). |
| No reading has arrived yet | Baymax says so and tells you to run the Shortcut. It never guesses a number. |
| The reading is more than 30 minutes old | Baymax says it may not match how you feel now. |
| A very fast (150+) or very slow (40 or below) reading, **and** you describe chest pain, fainting, dizziness, or trouble breathing | A fixed emergency message (call 911) is spoken. The model isn't consulted. |
| A fast or slow reading with **no** symptoms | Explained normally, not escalated. |

Readings are saved in plain text at `C:\Users\<you>\.baymax\vitals.jsonl`. Delete the file to erase them. Values outside 25 to 250 beats per minute are rejected as sensor glitches.

## Troubleshooting

- **The Shortcut says it couldn't connect.** Baymax has to be running, and Tailscale on your iPhone has to be connected. Open the Tailscale app to check.
- **"wrong or missing token"** means the token in the Shortcut doesn't match the file. Copy it again.
- **No reading is found.** Open the Health app and confirm the watch has recorded a heart rate recently. The watch only samples periodically at rest.
- **The Tailscale address doesn't work.** Tailscale must be switched on on the PC too, and HTTPS certificates enabled for your tailnet. See the README's "Phone and Tailscale" section.
