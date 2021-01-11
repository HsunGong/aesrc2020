#!/bin/bash

# Copyright 2020 Audio, Speech and Language Processing Group @ NWPU (Author: Xian Shi)
# Apache 2.0

AESRC_train=/mnt/lustre/sjtu/users/yzl23/work_dir/local_data/is20_accent_aispeech/is20_AESRC_train
AESRC_eval=/mnt/lustre/sjtu/users/yzl23/work_dir/local_data/is20_accent_aispeech/eval_data
LIBRISPEECH=/mnt/lustre/sjtu/home/yfc07/workdir/data/librispeech

data_dir=`pwd`/data
train_dir=$data_dir/accent_train # ${train_dir}/origin ->> original data
eval_dir=$data_dir/accent_test # ${eval_dir}/origin
librespeech_train_dir=$data_dir/librespeech_train

data=$2         # data transformed into kaldi format

stage=2
feature_cmd="run.pl"
nj=50

vocab_size=1000

if [ $stage -le 0 ];then
    echo "unzip and rename each accent aesrc2020"
    if [[ -w $AESRC_train && -r $AESRC_train/AESRC2020.zip ]]; then
        unzip -q $AESRC_train/AESRC2020.zip -d $AESRC_train/
    fi

    # Train Set
    mkdir -p ${train_dir}/origin
    ln -fns ${AESRC_train}/American\ English\ Speech\ Data ${train_dir}/origin/US
    ln -fns ${AESRC_train}/British\ English\ Speech\ Data ${train_dir}/origin/UK
    ln -fns ${AESRC_train}/Chinese\ Speaking\ English\ Speech\ Data ${train_dir}/origin/CHN 
    ln -fns ${AESRC_train}/Indian\ English\ Speech\ Data ${train_dir}/origin/IND 
    ln -fns ${AESRC_train}/Portuguese\ Speaking\ English\ Speech\ Data ${train_dir}/origin/PT 
    ln -fns ${AESRC_train}/Russian\ Speaking\ English\ Speech\ Data ${train_dir}/origin/RU 
    ln -fns ${AESRC_train}/Japanese\ Speaking\ English\ Speech\ Data ${train_dir}/origin/JPN 
    ln -fns ${AESRC_train}/Korean\ Speaking\ English\ Speech\ Data ${train_dir}/origin/KR

    # Test Set
    mkdir -p ${eval_dir}/origin
    ln -fns ${AESRC_eval} ${eval_dir}/origin

    # Librespeech
    # mkdir -p ${librespeech_train_dir}
    ln -fns ${LIBRISPEECH} ${librespeech_train_dir}

    # generate kaldi format data for all
    echo "Generating kaldi format data."
    for work_dir in ($train_dir $eval_dir); do
        find -L ${work_dir}/origin -name '*.wav' > ${work_dir}/wavpath
        
        awk -F'/' '{print $(NF-2)"-"$(NF-1)"-"$NF}' ${work_dir}/wavpath | \
            sed 's:\.wav::g' \
            > ${work_dir}/uttlist
        paste ${work_dir}/uttlist ${work_dir}/wavpath > ${work_dir}/wav.scp

        python local/tools/preprocess.py ${work_dir}/wav.scp ${work_dir}/trans ${work_dir}/utt2spk # faster than for in shell
        ./utils/utt2spk_to_spk2utt.pl ${work_dir}/utt2spk > ${work_dir}/spk2utt
    done
fi

exit 1

# clean transcription
if [ $stage -le 1 ];then
    echo "Cleaning transcription."
    tr '[a-z]' '[A-Z]' < $data/data_all/trans > $data/data_all/trans_upper
    
    # turn "." in specific abbreviations into "<m>" tag
    sed -i -e 's: MR\.: MR<m>:g' -e 's: MRS\.: MRS<m>:g' -e 's: MS\.: MS<m>:g' \
        -e 's:^MR\.:MR<m>:g' -e 's:^MRS\.:MRS<m>:g' -e 's:^MS\.:MS<m>:g' $data/data_all/trans_upper 
	# fix bug
    sed -i 's:^ST\.:STREET:g' $data/data_all/trans_upper 
    sed -i 's: ST\.: STREET:g' $data/data_all/trans_upper 
    # punctuation marks
    sed -i "s%,\|\.\|?\|!\|;\|-\|:\|,'\|\.'\|?'\|!'\| '% %g" $data/data_all/trans_upper
    sed -i 's:<m>:.:g' $data/data_all/trans_upper
    # blank
    sed -i 's:[ ][ ]*: :g' $data/data_all/trans_upper
    paste $data/data_all/uttlist $data/data_all/trans_upper > $data/data_all/text
fi

# extracting filter-bank features and cmvn
if [ $stage -le 4 ];then 
    ./utils/fix_data_dir.sh $data/data_all
    ./steps/make_fbank.sh --cmd $feature_cmd --nj $nj --fbank-config conf/fbank.conf $data/data_all $data/feats/log $data/feats/ark
    ./steps/compute_cmvn_stats.sh $data/data_all $data/feats/log $data/feats/ark # for kaldi 
fi

# divide development set for cross validation
if [ $stage -le 5 ];then 
    for i in US UK IND CHN JPN PT RU KR;do 
        ./utils/subset_data_dir.sh --spk-list local/files/cvlist/${i}_cv_spk $data/data_all $data/cv/$i 
        cat $data/cv/$i/feats.scp >> $data/cv.scp 
    done
    ./utils/filter_scp.pl --exclude $data/cv.scp $data/data_all/feats.scp > $data/train.scp 
    ./utils/subset_data_dir.sh --utt-list $data/train.scp $data/data_all $data/train
	./utils/subset_data_dir.sh --utt-list $data/cv.scp $data/data_all $data/cv_all
	compute-cmvn-stats scp:$data/train/feats.scp `pwd`/$data/train/dump_cmvn.ark # for espnet
    rm $data/cv.scp $data/train.scp 
fi


# generate label file and dump features for track2:E2E
if [ $stage -le 6 ];then 
    for i in US UK IND CHN JPN PT RU KR;do 
        local/tools/dump.sh --cmd $feature_cmd --nj 3 --do_delta false \
            $data/cv/$i/feats.scp $data/train/dump_cmvn.ark $data/cv/$i/dump/log $data/cv/$i/dump # for track2 e2e testing
    done 
    local/tools/dump.sh --cmd $feature_cmd --nj $nj  --do_delta false \
        $data/train/feats.scp $data/train/dump_cmvn.ark $data/train/dump/log $data/train/dump # for track2 e2e training
    # for track1, utterance-level CMVN is applied
    for data_set in train cv_all; do
        set_dir=$data/$data_set
        # hack to set utterance-level spk2utt & utt2spk
        awk '{printf "%s %s\n", $1, $1 }' $set_dir/text > $set_dir/spk2utt.utt
        cp $set_dir/spk2utt.utt $set_dir/utt2spk.utt
        compute-cmvn-stats --spk2utt=ark:$set_dir/spk2utt.utt scp:$set_dir/feats.scp \
            ark,scp:`pwd`/$set_dir/cmvn_utt.ark,$set_dir/cmvn_utt.scp
        local/tools/dump_spk_yzl23.sh --cmd slurm.pl --nj 48 \
            $set_dir/feats.scp $set_dir/cmvn_utt.scp \
            exp/dump_feats/$data_set $set_dir/dump_utt $set_dir/utt2spk.utt
    done
fi


# generate label file for track1
if [ $stage -le 7 ];then 
    for i in train cv_all;do 
        cut -f 1 $data/$i/text > $data/$i/uttlist 
        cut -d '-' -f 1 $data/$i/text | sed -e "s:^:<:g" -e "s:$:>:g" > $data/$i/accentlist
        paste $data/$i/uttlist $data/$i/accentlist > $data/$i/utt2accent 
        rm $data/$i/uttlist
		local/tools/data2json.sh --nj 20 --feat $data/$i/dump_utt/feats.scp --text $data/$i/utt2accent --oov 8 $data/$i local/files/ar.dict > $data/$i/ar.json
	done
fi    


# generate label file for track2 e2e 
if [ $stage -le 8 ];then 
	# goolgle sentence piece toolkit is used to train a bpe model and decode
	mkdir -p $data/bpe 
	mkdir -p $data/lang 
	# male sure you have installed sentencepiece successfully
	spm_train  \
		--input=$data/train/trans_upper \
		--model_prefix=$data/bpe/bpe_${vocab_size} \
		--vocab_size=$vocab_size \
		--character_coverage=1.0 \
		--model_type=unigram
	python local/tools/word_frequency.py $data/train/trans_upper 0 $data/bpe/train 
	cut -d ' ' -f 1 $data/bpe/train.enwf | awk '{if(NF==1)print $0}' > $data/bpe/wordlist.txt 
    spm_encode \
		--model=$data/bpe/bpe_${vocab_size}.model  \
		--output_format=piece < $data/bpe/wordlist.txt > $data/bpe/bpelist.txt 
	paste $data/bpe/wordlist.txt $data/bpe/bpelist.txt > $data/lang/lexicon.txt
	sed -i 's:▁ :▁:g' $data/lang/lexicon.txt 
	python local/tools/apply_lexicon.py $data/lang/lexicon.txt $data/train/text $data/train/utt2tokens "<unk>" $data/train/.warning $data/lang/units.txt 
	local/tools/data2json.sh --nj 20 --feat $data/train/dump/feats.scp --text $data/train/utt2tokens --oov 0 $data/train $data/lang/units.txt > $data/train/asr.json || exit 1;
	for i in US UK IND CHN JPN PT RU KR; do 
		# units.txt generate form cv set aborted  
		python local/tools/apply_lexicon.py $data/lang/lexicon.txt $data/cv/$i/text $data/cv/$i/utt2tokens "<unk>" $data/cv/$i/.warning $data/cv/${i}/.units.txt  || exit 1;
		local/tools/data2json.sh --nj 20 --feat $data/cv/$i/dump/feats.scp --text $data/cv/$i/utt2tokens --oov 0 $data/cv/$i $data/lang/units.txt > $data/cv/$i/asr.json
	done 

fi

echo "local/prepare_data.sh succeeded"
exit 0;
