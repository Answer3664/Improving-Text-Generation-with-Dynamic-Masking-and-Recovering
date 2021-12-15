#!bin/perl

use strict;
use warnings;

my $cur_dir = "/home/lzd/mt-exp/wmt14-en2de";
my $workspace = "$cur_dir/bert-dynamic-ende";
my $gpu_id = "1";
my $python3 = "/home/lzd/anaconda3/bin/python3";
my $t2t_dir = "/home/lzd/transmodel/bert-dynamic-denoise";
my $multi_bleu = "/home/lzd/script/multi-bleu.perl";

my $src = "en";
my $tgt = "de";

my $step0 = 0; #move data
my $step1 = 0; #prepare_data
my $step2 = 0; #training
my $step3 = 0; #tuning
my $step4 = 0; #test
my $step5 = 0; #infer with ensemble model

my $data_dir = "$workspace/data";
makedir("$data_dir");
my $model_dir = "$workspace/model";
makedir("$model_dir");
my $prep = "$workspace/wmt14";
makedir("$prep");
my $train_test_data_dir = "$prep";
my $tmp_dir = "$workspace/tmp";
makedir("$tmp_dir");
my $valid = "valid";
my @test_sets = ("test");

my $cmd;

my $data_from = "/home/lzd/corpus/wmt14";

if ($step0){
  for my $set ("train") {
    $cmd = "ln -s $data_from/$set.$src $tmp_dir/; ln -s $data_from/$set.$tgt $tmp_dir/";
    run("$cmd");
    $cmd = "ln -s $tmp_dir/$set.$src $prep/$set.$src";
    run("$cmd");
    $cmd = "ln -s $tmp_dir/$set.$tgt $prep/$set.$tgt";
    run("$cmd");
  }
  for my $set ($valid) {
    $cmd = "ln -s $data_from/$set.$src $tmp_dir/; ln -s $data_from/$set.$tgt $tmp_dir/ ln -s $data_from/$set.tok.$tgt $tmp_dir/";
    run("$cmd");
    $cmd = "cp $tmp_dir/$set.$src $prep/valid.$src";
    run("$cmd");
    $cmd = "cp $tmp_dir/$set.$tgt $prep/valid.$tgt";
    run("$cmd");
    $cmd = "cp $tmp_dir/$set.tok.$tgt $prep/valid.tok.$tgt";
    run("$cmd");


  }
  for my $set (@test_sets) {
    $cmd = "ln -s $data_from/$set.$src $tmp_dir/";
    run("$cmd");
    $cmd = "ln -s $data_from/$set.tok.$tgt $tmp_dir/$set.tok.$tgt";
    run("$cmd");
    $cmd = "ln -s $tmp_dir/$set.tok.$tgt $prep/";
    run("$cmd");
    $cmd = "ln -s $tmp_dir/$set.$src $prep/$set.$src";
    run("$cmd");
  }

}

if ($step1) {
  die "FILE NOT FOUND" unless (-e "$train_test_data_dir/train.$src" and -e "$train_test_data_dir/train.$tgt" and -e "$train_test_data_dir/valid.$src" and -e "$train_test_data_dir/valid.$tgt");
  $cmd = "$python3 $t2t_dir/preprocess.py -train_src $train_test_data_dir/train.$src -train_tgt $train_test_data_dir/train.$tgt -valid_src $train_test_data_dir/valid.$src -valid_tgt $train_test_data_dir/valid.$tgt -save_data $data_dir/wmt14 -share_vocab -src_vocab_size 40000 -tgt_vocab_size 40000 -src_seq_length 1500 -tgt_seq_length 1500 1>$workspace/prepare.log 2>&1";
  run("$cmd");
}


if ($step2) {
  my $world_size = 1;
  my $gpu_ranks = "0";
  my @gpus = split(/,/, $gpu_id);
  $world_size = scalar(@gpus);
  for (my $i = 1; $i < scalar(@gpus); $i++) {
    $gpu_ranks = $gpu_ranks . " $i";
  }
  my $accum_count = 2;
  my $batch_size = 8192;
  my $select_layer = 4; #[0,1,2,3,4,5]
  my $alpha = 0.6;
  my $bert_dec_layers = 1;
  $cmd = "CUDA_VISIBLE_DEVICES=$gpu_id $python3 $t2t_dir/train.py -bert_dec_layers $bert_dec_layers -alpha $alpha -select_layer $select_layer -data $data_dir/wmt14 -save_model $model_dir/model -world_size $world_size -gpu_ranks $gpu_ranks  -accum_count $accum_count -save_checkpoint_steps 5000 -valid_steps 5000 -report_every 20 -keep_checkpoint 50 -seed 3435 -train_steps 300000 -warmup_steps 16000 --share_embeddings --share_decoder_embeddings --position_encoding --optim adam -adam_beta1 0.9 -adam_beta2 0.998 -decay_method noam -learning_rate 2.0 -max_grad_norm 0.0 -batch_size $batch_size -batch_type tokens -normalization tokens -dropout 0.1 -label_smoothing 0.1 -max_generator_batches 128 -param_init 0.0 -param_init_glorot -valid_batch_size 8 1> $model_dir/train.log 2>&1";
  run("$cmd");
}

my @data_sets = ("valid"); 

if ($step3) {
  @data_sets = ("valid");
  foreach my $set (@data_sets) {
    my $start_iter = 150000;
    my $end_iter = 300000;
    my $iter = $start_iter;
    while ($iter <= $end_iter) {
      if (-e "$model_dir/model_step_$iter.pt" and !-e "$model_dir/$iter/$set.tran") {
        print("Decoding using $model_dir/model_step_$iter.pt\n");
        makedir("$model_dir/$iter");
        run("touch $model_dir/$iter/$set.tran");
        $cmd = "CUDA_VISIBLE_DEVICES=$gpu_id $python3 $t2t_dir/translate.py -model $model_dir/model_step_$iter.pt -src $train_test_data_dir/$set.$src -output $model_dir/$iter/$set.tran -beam_size 5 -gpu 1 1>$model_dir/$iter/$set.log 2>&1";
        run("$cmd");

        $cmd = "sed -i \"s/@@ //g\" $model_dir/$iter/$set.tran";
        run("$cmd");
        eval_multibleu("$model_dir/$iter/$set.tran", "$train_test_data_dir/$set.tok.$tgt");
        #eval_fastscore("$model_dir/$iter/$set.tran", "$tmp_dir/$set.$tgt");
        #eval_chrf("$model_dir/$iter/$set.tran", "$tmp_dir/$set.$tgt");
        }
        $iter += 5000;
    }
  }
}

if ($step4) {
  my $best_iteration = "290000";
  foreach my $set (@test_sets) {
    my $iter = $best_iteration;
    makedir("$model_dir/$iter");
    $cmd = "CUDA_VISIBLE_DEVICES=$gpu_id $python3 $t2t_dir/translate.py -model $model_dir/model_step_$iter.pt -src $train_test_data_dir/$set.$src -output $model_dir/$iter/$set.tran -beam_size 5 -gpu 1 1>$model_dir/$iter/$set.log 2>&1";
    run("$cmd");

    $cmd = "sed -i \"s/@@ //g\" $model_dir/$iter/$set.tran";
    run("$cmd");
    eval_multibleu("$model_dir/$iter/$set.tran", "$train_test_data_dir/$set.tok.$tgt");
    #eval_fastscore("$model_dir/$iter/$set.tran", "$tmp_dir/$set.$tgt");
  }
} 


sub eval_multibleu {
  my $tran = shift;
  my $ref = shift;
  my $mycmd = "perl $multi_bleu $ref < $tran > $tran.evalmb";
  run("$mycmd");
}

sub makedir {
  my $dir = shift;
  my $mycmd = "mkdir -p $dir";
  run("$mycmd") unless -e "$dir";
}

sub run {
  my $mycmd = shift;
  print "$mycmd\n";
  system("$mycmd");
}






























