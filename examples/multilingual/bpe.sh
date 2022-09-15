SPM=../../scripts/spm_encode.py
MODEL=mbart.cc25.v2/sentence.bpe.model
DATA=data_scripts/data/ML50/raw
TRAIN=train.$1-$2
VALID=valid.$1-$2
TEST=test.$1-$2
SRC=$1
TGT=$2

python ${SPM} --model=${MODEL} < ${DATA}/${TRAIN}.${SRC} > ${DATA}/${TRAIN}.spm.${SRC} &
python ${SPM} --model=${MODEL} < ${DATA}/${TRAIN}.${TGT} > ${DATA}/${TRAIN}.spm.${TGT} &
python ${SPM} --model=${MODEL} < ${DATA}/${VALID}.${SRC} > ${DATA}/${VALID}.spm.${SRC} &
python ${SPM} --model=${MODEL} < ${DATA}/${VALID}.${TGT} > ${DATA}/${VALID}.spm.${TGT} &
python ${SPM} --model=${MODEL} < ${DATA}/${TEST}.${SRC} > ${DATA}/${TEST}.spm.${SRC} &
python ${SPM} --model=${MODEL} < ${DATA}/${TEST}.${TGT} > ${DATA}/${TEST}.spm.${TGT} &