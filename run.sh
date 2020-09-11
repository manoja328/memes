#for training
#mmf_run config=projects/hateful_memes/configs/visual_bert/from_coco.yaml model=visual_bert dataset=hateful_memes

#mmf_run config=projects/hateful_memes/configs/visual_bert/from_coco.yaml \
#      model=visual_bert \
#       dataset=hateful_memes \
#      run_type=val checkpoint.resume_file=$MODEL checkpoint.resume_pretrained=False



#from mmf.common.registry import registry
#model_cls = registry.get_model_class("visual_bert")
#model = model_cls.from_pretrained("visual_bert.finetuned.hateful_memes.from_coco")

#MODEL="/home/manoj/.cache/torch/mmf/data/models/visual_bert.finetuned.hateful_memes.from_coco/model.pth"


#Evaluation on submission
MMF_USER_DIR="." mmf_predict config=projects/hateful_memes/configs/visual_bert/from_coco.yaml model=visual_bert dataset=hateful_memes \
run_type=test checkpoint.resume_zoo=visual_bert.finetuned.hateful_memes.from_coco checkpoint.resume_pretrained=False

#Eval on val set
MMF_USER_DIR="." mmf_predict config=projects/hateful_memes/configs/visual_bert/from_coco.yaml model=visual_bert dataset=hateful_memes \
run_type=val checkpoint.resume_zoo=visual_bert.finetuned.hateful_memes.from_coco checkpoint.resume_pretrained=False