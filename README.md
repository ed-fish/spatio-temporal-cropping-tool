# embeddingator
Collect Multi-Modal embeddings from video clips automatically

Easy to use multi-modal embedding extractor for video clips - specifically for contrastive learning strategies with multiple augmentations. 

Current models include:
  - Image classification: Resnet50 (Imagenet) 1 x 2048 embedding 
  - Location classification: Resnet50 (Places365) 1 x 2048 embedding
  - Action recognition: Resnet18 (Kinetics400) 1 x 516 embedding
  - Depth perception: intelISL (MiDaS)  1 x 2048 embedding
  
More models being added shortly. 

Update the config.yml with your requirements before running. 

This code is setup for a specific file structure - for a project related to movies. As such you will need to adapt the code to fit your individual case. 

In the future I will be updating the code to make it more flexible, if you use the code and make it more genralised please submit a PR. 

Check back in a few weeks for updates. 
