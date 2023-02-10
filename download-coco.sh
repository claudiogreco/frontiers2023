curl http://images.cocodataset.org/annotations/annotations_trainval2014.zip -o "./data/preprocess/annotations_trainval2014.zip"
curl http://images.cocodataset.org/zips/val2014.zip -o "./data/preprocess/val2014.zip"

unzip ./data/preprocess/annotations_trainval2014.zip -d ./data/preprocess
unzip ./data/preprocess/val2014.zip -d ./data/preprocess

rm ./data/preprocess/annotations_trainval2014.zip
rm ./data/preprocess/val2014.zip

mv ./data/preprocess/val2014 ./data/preprocess/val2014_raw

python ./data/preprocess/resize.py

rm -r ./data/preprocess/val2014_raw
