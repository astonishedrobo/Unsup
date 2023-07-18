N=13000  # Number of files to copy
source_folder="/home/soumyajit/ADEChallengeData2016/images/training"
destination_folder="/home/soumyajit/ADEChallengeData2016/images/finetune"

ls -1 "$source_folder" | sort | tail -n $N | xargs -I {} mv "$source_folder/{}" "$destination_folder/"
