# Dataset

Audio recordings of coughs and other sounds (laughing, clapping, knocking, speech, crowd noise), with timestamps for each event.

## Audio recordings

| Folder | Source |
|---|---|
| `audio-recordings/cough-recordings` | Coughs recorded by me, friends and colleagues |
| `audio-recordings/cough-recordings-internet` | Coughs collected online |
| `audio-recordings/coughvid-dry-coughs` | Dry coughs from the COUGHVID dataset, labelled by experts |
| `audio-recordings/coughvid-wet-coughs` | Wet coughs from the COUGHVID dataset, labelled by experts |
| `audio-recordings/other-recordings` | Other sounds, self-recorded |
| `audio-recordings/other-recordings-internet` | Other sounds collected online |

## Timestamps

`audio-timestamps/` has one `.txt` file per recording, in subfolders that match the recording folders above (e.g. `cough-timestamps` for `cough-recordings`). Each line holds a start and an end time in seconds. For coughs, these mark each cough event; for other sounds, segments were chosen more randomly to give the classifiers varied examples.

## Metadata

`cough-recordings.csv` lists the file name, number of coughs and label for each self-recorded cough recording.
