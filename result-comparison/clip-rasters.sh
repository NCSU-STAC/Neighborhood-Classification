# extract how many ever features are in the script
# First argument - shape file
# second argument - Raster 
# Hardcoded the target resolution in the code - change if it is different
# @author: Krishna Karthik Gadiraju

outfile="/scratch/slums/bl-slums/result-comparison/edge_density_extract_415"
extension=".tif"
for i in 1 2 3 4
do
gdalwarp -q -cutline "$1" -crop_to_cutline -tr 0.499284950117 0.499231422086 -of GTiff "$2" $outfile"_"$i$extension -cwhere 'id'=$i
done 
