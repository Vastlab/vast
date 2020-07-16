rm -rf /tmp/testingssh.txt
servers=(
    "ironman"
    "pepper"
    "reddwarf"
    "deathstar"
)
for i in "${servers[@]}"; do
    echo "trying $i"
    echo "trying $i" >> /tmp/testingssh.txt
    ssh adhamija@$i.vast.uccs.edu "/home/adhamija/anaconda3/envs/dali/bin/python -c 'import torch;print(torch.__version__)' 2>&1" >> /tmp/testingssh.txt
done
