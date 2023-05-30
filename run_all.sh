# echo "Running all configs"

# echo "Starting EmoCNN"
# python3 emotion_recognition.py --config configs/config_emo_adaptive_lr_augmented.json > log.txt | mv log.txt $(ls -td -- outputs/*/ | head -n 1)
# echo "Done 1"
# python3 emotion_recognition.py --config configs/config_emo_adaptive_lr_not_augmented.json > log.txt | mv log.txt $(ls -td -- outputs/*/ | head -n 1)
# echo "Done 2"
# python3 emotion_recognition.py --config configs/config_emo_not_adaptive_lr_augmented.json > log.txt | mv log.txt $(ls -td -- outputs/*/ | head -n 1)
# echo "Done 3"
# echo "Done EmoCNN"

# echo "Starting Emo2CNN"
# python3 emotion_recognition.py --config configs/config_emo2_adaptive_lr_augmented.json > log.txt | mv log.txt $(ls -td -- outputs/*/ | head -n 1)
# echo "Done 1"
# python3 emotion_recognition.py --config configs/config_emo2_adaptive_lr_not_augmented.json > log.txt | mv log.txt $(ls -td -- outputs/*/ | head -n 1)
# echo "Done 2"
# python3 emotion_recognition.py --config configs/config_emo2_not_adaptive_lr_augmented.json > log.txt | mv log.txt $(ls -td -- outputs/*/ | head -n 1)
# echo "Done 3"
# echo "Done Emo2CNN"

# echo "Starting Emo3CNN"
# python3 emotion_recognition.py --config configs/config_emo3_adaptive_lr_augmented.json > log.txt | mv log.txt $(ls -td -- outputs/*/ | head -n 1)
# echo "Done 1"
# python3 emotion_recognition.py --config configs/config_emo3_adaptive_lr_not_augmented.json > log.txt | mv log.txt $(ls -td -- outputs/*/ | head -n 1)
# echo "Done 2"
# python3 emotion_recognition.py --config configs/config_emo3_not_adaptive_lr_augmented.json > log.txt | mv log.txt $(ls -td -- outputs/*/ | head -n 1)
# echo "Done 3"
# echo "Done Emo3CNN"

echo "Starting Emo4CNN"
python3 emotion_recognition.py --config configs/config_emo4_adaptive_lr_augmented.json > log.txt | mv log.txt $(ls -td -- outputs/*/ | head -n 1)
echo "Done 1"
python3 emotion_recognition.py --config configs/config_emo4_adaptive_lr_not_augmented.json > log.txt | mv log.txt $(ls -td -- outputs/*/ | head -n 1)
echo "Done 2"
python3 emotion_recognition.py --config configs/config_emo4_not_adaptive_lr_augmented.json > log.txt | mv log.txt $(ls -td -- outputs/*/ | head -n 1)
echo "Done 3"
echo "Done Emo4CNN"

echo "Starting Emo5CNN"
python3 emotion_recognition.py --config configs/config_emo5_adaptive_lr_augmented.json > log.txt | mv log.txt $(ls -td -- outputs/*/ | head -n 1)
echo "Done 1"
python3 emotion_recognition.py --config configs/config_emo5_adaptive_lr_not_augmented.json > log.txt | mv log.txt $(ls -td -- outputs/*/ | head -n 1)
echo "Done 2"
python3 emotion_recognition.py --config configs/config_emo5_not_adaptive_lr_augmented.json > log.txt | mv log.txt $(ls -td -- outputs/*/ | head -n 1)
echo "Done 3"
echo "Done Emo5CNN"

echo "Starting Emo6CNN"
python3 emotion_recognition.py --config configs/config_emo6_adaptive_lr_augmented.json > log.txt | mv log.txt $(ls -td -- outputs/*/ | head -n 1)
echo "Done Emo6CNN"

echo "Done with all configs"