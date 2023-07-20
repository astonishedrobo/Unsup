N=2000  # Number of files to copy
source_folder="/mnt/efs/Data/random_sampled_ade_train/annotations/"
destination_folder="/mnt/efs/Data/random_sampled_ade_val/annotations/"

ls -1 "$source_folder" | sort | tail -n $N | xargs -I {} mv "$source_folder/{}" "$destination_folder/"
