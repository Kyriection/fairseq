SPM=../../scripts/spm_encode.py
MODEL=mbart.cc25.v2/sentence.bpe.model
DATA=data_scripts/data/indic_languages_corpus/indic_languages_corpus/bilingual/$1-$2
TRAIN=train
VALID=dev
TEST=test
SRC=$1
TGT=$2

gunzip ${DATA}/${TRAIN}.${SRC}.gz
gunzip ${DATA}/${TRAIN}.${TGT}.gz
gunzip ${DATA}/${VALID}.${SRC}.gz
gunzip ${DATA}/${VALID}.${TGT}.gz
gunzip ${DATA}/${TEST}.${SRC}.gz
gunzip ${DATA}/${TEST}.${TGT}.gz

python ${SPM} --model=${MODEL} < ${DATA}/${TRAIN}.${SRC} > ${DATA}/${TRAIN}.spm.${SRC} &
python ${SPM} --model=${MODEL} < ${DATA}/${TRAIN}.${TGT} > ${DATA}/${TRAIN}.spm.${TGT} &
python ${SPM} --model=${MODEL} < ${DATA}/${VALID}.${SRC} > ${DATA}/${VALID}.spm.${SRC} &
python ${SPM} --model=${MODEL} < ${DATA}/${VALID}.${TGT} > ${DATA}/${VALID}.spm.${TGT} &
python ${SPM} --model=${MODEL} < ${DATA}/${TEST}.${SRC} > ${DATA}/${TEST}.spm.${SRC} &
python ${SPM} --model=${MODEL} < ${DATA}/${TEST}.${TGT} > ${DATA}/${TEST}.spm.${TGT} &