### Jan

### Martin
. ./pm_martin.source
`pip install -e .` for datacircuits

cd stable

1. `./submit_multXY_job.py --numQaddr 3  --numShot 10_000  --numSample 50  -E`
2. `./postproc_multXY.py  --basePath  $basePath  --expName   ideal_7100ad   -p a c   -Y
`


curl "https://api.ionq.co/v0.3/jobs/019832f4-e6a5-704d-a31b-36553ea45149/results" \
  -H "Authorization: apiKey xmWCkOA5c9tZj3PTRFKwmzPR7jFAo7gI"

### dry Run
./submit_multXY_job.py --numQaddr 3  --numShot 20_000  --numSample 50  -E -b ionq -d --id 01983079-f8d0-764a-9f62-d8f074678e77
basePath=/pscratch/sd/g/gzquse/quantum-transformer/stable/out
./retrieve_ionq_job.py  --basePath  $basePath --expName   ionq_835858
./postproc_multXY.py  --basePath  $basePath  --expName   ionq_835858   -p a c   -Y