#!/bin/bash
cd `dirname "$0"`
cd DATA
mkdir -p "./raw_data/fg"
mkdir -p "./raw_data/bg"

for ((i=1; i<=10; ++i)); do
    tarname=`printf "data%02d.tar.gz" $i`
    tar -zxvf "$tarname" -C "./raw_data/fg" --strip-components=1
done

unzip backgrounds.zip -d "./raw_data/bg"
unzip processedData.zip -d "./processedData"
uniq "./processedData/z.txt" > "./processedData/z_uniq.txt"

cd ..

echo "Generating numpy objects..."
sync && python preprocess_id.py

