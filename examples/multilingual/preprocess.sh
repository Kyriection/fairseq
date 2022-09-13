SRC=my
TGT=en
DATA=$1
TRAIN=train.alt
VALID=dev.alt
TEST=test.alt
DICT=mbart.cc25.v2/dict.txt
DEST=data 
NAME=my_en
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