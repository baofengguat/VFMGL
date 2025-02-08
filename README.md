In this study, we used the DINOv2 series vision foundation model on an Ubuntu system for experiments. Below is a simple demo guide:
1、Multi-center Data Download
	Example data: Camelyon17 dataset (classification task) and Nuclei dataset (segmentation task) can be downloaded from the following link:
	https://pan.baidu.com/s/1urddOKCsQ-KQr786eUjL7A?pwd=u1br
	extraction code: u1br
2、Pre-trained Weights for the Vision Foundation Model
	The compressed package includes a DINOv2 model parameter.
	More model weight files can be downloaded from the following website:
	https://github.com/facebookresearch/dinov2
3、Unzip the downloaded multi-center data and place it in the ./data/ folder.
4、Install and Activate the Runtime Environment (Anaconda) and Install Dependencies:
	wget https://repo.anaconda.com/archive/Anaconda3-2023.03-Linux-x86_64.sh
	bash Anaconda3-2023.03-Linux-x86_64.sh
	source ~/.bashrc
	conda create --name new_env python=3.9
	conda activate new_env
	cd ./code
	pip install -r requirements.txt
5、Run demo:
	# for classification task:
	python demo_classification.py
	# for segmentation task:
	python demo_segment.py
To facilitate your reading, we have added necessary code comments in the scripts. We have also preset the hyperparameters for model training, allowing you to run the demo directly. If you wish to set your own hyperparameters, you can do so according to the parameter descriptions.
If you want to shorten the model training time, and if CPU and GPU resources allow, you can set a larger batch_size or adjust the num_workers quantity in the get_args function in the demo files. Conversely, you can reduce the batch_size and num_works to free up more computing resources.
6、Output:
	Model files and results will be saved in ./model_save.
7、Model Evaluation:
	# for classification task:
	python classification_metric_compute.py
	# for segmentation task:
	python segmentation_metric_compute.py
The trained model parameters mentioned in the paper are saved in the ./model_weight folder. If you wish to reproduce the results in the paper, you can use these scripts to load the model weights and data for validation.

Thank you for your attention!

# VFMGL
