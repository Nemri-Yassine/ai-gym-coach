# Week 2 presentation · what to say (about 12 minutes, 17 slides)

Open `reports/validation2/AI_Workout_Coach_Week2.html` in Chrome; keep the `demo/` folder next to it.
**→ / ←** change slide · **F** fullscreen · **double-click** a video to make it fullscreen.

| # | Slide | Say (short, simple) |
|---|---|---|
| 1 | Title | "This week we asked one question: do our models still work on real gym videos, with our phone, on our bodies?" |
| 2 | The question | "Model 1 and Model 2 were trained on public datasets. RQ4 measures how much we lose on a real phone camera. Every week-2 step is done." |
| 3 | Pipeline | "Every set goes through this chain: iPhone → skeleton → which exercise → reps → form. The watch sends heart rate on the same clock. Kafka and Spark handle the streams and the rep table. All on a laptop CPU." |
| 4 | Station | "We built this ourselves: the iPhone streams each set live to the laptop, the watch sends heart rate over Bluetooth. 30 fps on every set, nothing lost. Heart rate is our weak point: about half the time covered." |
| 5 | Our data | "Two of us, 20 sets, 9 exercises. The models never saw these videos. We pressed the wrong button 3 times; our written notes are the truth." |
| 6 | **Demo** | Let the montage play. "Left is our video, right is what the model sees live: exercise, confidence, reps, heart rate, and the truth in grey." |
| 7 | Results | "14 of 20 sets right without any training on our data. Perfect on 5 exercises, wrong on 4. On public data the same model is around 80%, so we lose about 10 points on our phone." |
| 8 | It works | "Two people, two exercises, correct live." |
| 9 | Our mistake | "Here we pressed 'squats' but did push-ups. The model said push-ups. It can check the labels we type." |
| 10 | **Why it fails** | Click the 3 buttons. "Triceps: ours is a cable push-down from the side; the dataset only has overhead extensions. Lateral raises look like a jumping jack. One-arm rows from the side look like a push-up." |
| 11 | Proof | "This plot shows our movement against the training data. The failures are systematic, not random: a different variant or a different camera view." |
| 12 | The fix | "We adapted the model on one person and tested on the other. Yassine went from 75% to 92%. For Iheb it didn't change. With 8 and 12 sets this is a first signal, not a proof." |
| 13 | Reps + HR | "Squats are counted exactly. Push-ups are off by about 2. The true counts are being re-checked by hand, so these are provisional. Heart rate rises during each set, as expected." |
| 14 | **Form demo** | Let it play. "This is Model 2 on a person it never saw: left a correct squat, right knees inward. It flags the mistake. Same for lunges and plank. And here is a miss: we show it on purpose." |
| 15 | **Form on a real video** | "Now the same model on our own real video. This squat is correct, but it says 'front bent' on every rep. It learned from motion-capture skeletons; a phone skeleton looks different. That's why the live coach hides form until we calibrate it on our own recordings in week 3." |
| 16 | Form + big data | "The form model works in the lab, but on our correct squats it says 'bad form' every time. So we hide it in the demo until it's calibrated. Kafka streams at 30 messages per second; Spark builds the rep table." |
| 17 | Lessons | "The models work where our gym looks like their data. Where it doesn't, we know why and how to fix it. Week 3: longer sessions, fatigue model on our data, more people." |

## Hard questions

- **"Why only 70%?"** The dataset uses other variants and a front camera. We showed the reason with a figure, and that adaptation helps (75 → 92%).
- **"Isn't 20 sets too small?"** Yes. We give confidence intervals (48–85%) and say it's a first signal. Week 3 adds sessions.
- **"Did you tune on your test videos?"** No. Our videos are test only. The adaptation trains on one person and tests on the other.
- **"Who checks the rep counts?"** We re-count by hand from the videos (`scripts/review_reps.py`). Until then the rep numbers are marked provisional.
- **"Can Model 2 detect a mistake on a real video?"** Not yet reliably: on our correct squats it flags every rep. On its own data it works on a person it never saw (EC3D demo). Calibrating on our own recorded mistakes is the week-3 fix.
- **"Why hide the form model?"** It gives P(correct) = 0.06 on squats that were correct. Showing it would mislead the user.
