servers=(
    "ironman"
    "pepper"
)
for i in "${servers[@]}"; do
    echo "Copying to $i"
    rsync -zarvh /scratch/datasets/SAILON/ adhamija@$i.vast.uccs.edu:/scratch/datasets/SAILON_New/
done
echo "Copying to Kato"
rsync -zarvh /scratch/datasets/SAILON/ adhamija@kato.vast.uccs.edu:/store2/SAILON_New/
