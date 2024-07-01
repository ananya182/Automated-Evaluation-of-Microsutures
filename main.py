import sys
from part1 import main as part1_main
from part2 import main as part2_main

part_id = int(sys.argv[1])
img_dir = sys.argv[2]
output_csv = sys.argv[3]

if part_id == 1:
    part1_main(img_dir, output_csv)
else:
    part2_main(img_dir, output_csv)

    
