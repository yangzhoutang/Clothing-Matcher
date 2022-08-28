# Clothing-Matcher
Use a combination of CV and fashion theory, provide a clothing matcher to help people decide what to dress。

### This project is contributed by *Yangzhou Tang*, *Kaihang Zhao*, *Fan Li* and *Chenjia Guo*.

## Techinal details

* **U2NET** : This project uses an amazing [U2NET](https://arxiv.org/abs/2005.09007) as a deep learning model. Instead of having 1 channel output from u2net for typical salient object detection task it outputs 4 channels each respresting upper body cloth, lower body cloth, fully body cloth and background. Only categorical cross-entropy loss is used for a given version of the checkpoint.

* **Dataset** : U2net is trained on 45k images [iMaterialist (Fashion) 2019 at FGVC6](https://www.kaggle.com/c/imaterialist-fashion-2019-FGVC6/data) dataset. To reduce complexity, I have clubbed the original 42 categories from dataset labels into 3 categories (upper body, lower body and full body). All images are resized into square `¯\_(ツ)_/¯` 768 x 768 px for training. (This experiment was conducted with 768 px but around 384 px will work fine too if one is retraining on another dataset).
