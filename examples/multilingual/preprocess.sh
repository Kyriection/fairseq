SRC=$1
TGT=$2
DATA=dataset
TRAIN=opus.$1-$2-train
VALID=opus.$1-$2-dev
TEST=opus.$1-$2-test
DICT=mbart.cc25.v2/dict.txt
DEST=data 
NAME=$1_$2
fairseq-preprocess \
    --source-lang ${SRC} \
    --target-lang ${TGT} \
    --trainpref ${DATA}/${TRAIN}.spm \
    --validpref ${DATA}/${VALID}.spm \
    --testpref ${DATA}/${TEST}.spm \
    --destdir ${DEST}/${NAME} \
    --thresholdtgt 0 \
    --thresholdsrc 0 \
    --srcdict ${DICT} \
    --tgtdict ${DICT} \
    --workers 70