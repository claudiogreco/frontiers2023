# frontiers2023
Source code related to: Greco C., Bagade D., Le T. D., Bernardi R. (2023) She adapts to her student: An expert pragmatic speaker tailoring her referring expressions to the layman listener. In: Frontiers 2023. (https://doi.org/10.3389/frai.2023.1017204)

Please, cite:
```
@ARTICLE{10.3389/frai.2023.1017204,
  
AUTHOR={Greco, Claudio and Bagade, Diksha and Le, Dieu-Thu and Bernardi, Raffaella},   
	 
TITLE={She adapts to her student: An expert pragmatic speaker tailoring her referring expressions to the Layman listener},      
	
JOURNAL={Frontiers in Artificial Intelligence},      
	
VOLUME={6},           
	
YEAR={2023},      
	  
URL={https://www.frontiersin.org/articles/10.3389/frai.2023.1017204},       
	
DOI={10.3389/frai.2023.1017204},      
	
ISSN={2624-8212},   
   
ABSTRACT={Communication is a dynamic process through which interlocutors adapt to each other. In the development of conversational agents, this core aspect has been put aside for several years since the main challenge was to obtain conversational neural models able to produce utterances and dialogues that at least at the surface level are human-like. Now that this milestone has been achieved, the importance of paying attention to the dynamic and adaptive interactive aspects of language has been advocated in several position papers. In this paper, we focus on how a Speaker adapts to an interlocutor with different background knowledge. Our models undergo a pre-training phase, through which they acquire grounded knowledge by learning to describe an image, and an adaptive phase through which a Speaker and a Listener play a repeated reference game. Using a similar setting, previous studies focus on how conversational models create new conventions; we are interested, instead, in studying whether the Speaker learns from the Listener's mistakes to adapt to his background knowledge. We evaluate models based on Rational Speech Act (RSA), a likelihood loss, and a combination of the two. We show that RSA could indeed work as a backbone to drive the Speaker toward the Listener: in the combined model, apart from the improved Listener's accuracy, the language generated by the Speaker features the changes that signal adaptation to the Listener's background knowledge. Specifically, captions to unknown object categories contain more adjectives and less direct reference to the unknown objects.}
}
```

## Abstract
Communication is a dynamic process through which interlocutors adapt to each other. In the development of conversational agents, this core aspect has been put aside for several years since the main challenge was to obtain conversational neural models able to produce utterances and dialogues that at least at the surface level are human-like. Now that this milestone has been achieved, the importance of paying attention to the dynamic and adaptive interactive aspects of language has been advocated in several position papers. In this paper, we focus on how a Speaker adapts to an interlocutor with di  erent background knowledge. Our models undergo a pre-training phase, through which they acquire grounded knowledge by learning to describe an image, and an adaptive phase through which a Speaker and a Listener play a repeated reference game. Using a similar setting, previous studies focus on how conversational models create new conventions; we are interested, instead, in studying whether the Speaker learns from the Listener’s mistakes to adapt to his background knowledge. We evaluate models based on Rational Speech Act (RSA), a likelihood loss, and a combination of the two. We show that RSA could indeed work as a backbone to drive the Speaker toward the Listener: in the combined model, apart from the improved Listener’s accuracy, the language generated by the Speaker features the changes that signal adaptation to the Listener’s background knowledge. Specifically, captions to unknown object categories contain more adjectives and less direct reference to the unknown objects.

## Structure
The source code integrates three repositories:
* https://github.com/hawkrobe/continual-adaptation: we used their listener architecture and context loader and we developed our scripts for the interactive setup between speaker and listener models starting from the scripts they wrote for their interactive setup between speaker / listener models and humans. The "models" folder in our repository includes files from the "models" folder in their repository. The files we edited from their repository show the "custom_" prefix in their name.
* https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning: in order to train the listener architecture with our pre-training dataset, we used their training code. The files from their repository are in "image_captioning". We also used the model pre-trained by them on MS-COCO for our expert listener, whose weights are stored in the "data/models/listener" folder.
* https://github.com/reubenharry/Recurrent-RSA: we used their speaker model which implements Rational Speech Act (RSA) for the generation of pragmatic captions. The files from their repository are stored in the "bayesian_agents" and "recurrent_rsa" folders.

## Environment setup
Run the following commands:
```
virtualev venv -p python3
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Data setup
Execute the following instructions:

1. Download the data (contexts, listener pre-training data, and pre-trained models) from the following link:
https://drive.google.com/drive/folders/1R77hFZHbJHbRINfcBa7p0zzvAUtRlvqb?usp=sharing
(The data can also be downloaded by installing gdown (pip install gdown) and running gdown 12CUXJF-d_8uRtW5FhCmRQCGvqUoseXiL)
Alternatively, you can download data from Zenodo at the link: https://zenodo.org/record/7750153#.ZCawqS9By9k
2. Place the downloaded "data" folder in the root directory of the repository
3. Run the command wget http://images.cocodataset.org/zips/val2014.zip
4. Extract the files in the "val2014.zip" archive to the "data/preprocess/val2014" folder
5. Run the command wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
6. Extract the files in the "annotations_trainval2014.zip" archive to the "data/preprocess/annotations" folder

### Contexts
- The generated contexts are contained in the file "data/preprocess/2_unknown_2_known.json"
- The sampled contexts used for our experiments are contained in the file "data/preprocess/adaptation_2_unknown_2_known.json"

### Reports
The generated reports showing the speaker adaptation during the interaction with the expert and layman listeners are contained in the folder "data/reports".

### Pre-trained models
The pre-trained speaker and listener models are contained in the folders "data/models/speaker" and "data/models/listener", respectively.

### Pre-training data
The data used to pre-train the layman listener is contained in the file "data/preprocess/captions_train2014_wrt_2_unknown_2_known.json".

## Usage
### Interactive setup
In order to run the interactive setup, run:
```
python --speaker_loss=<SPEAKER_LOSS> --speaker_reset_after=<SPEAKER_RESET_AFTER> --listener_encoder_path=<LISTENER_ENCODER_PATH> --listener_decoder_path=<LISTENER_DECODER_PATH> --report_path=<REPORT_PATH> speaker_adaptation_to_listener_rsa.py
```
where:
- <SPEAKER_LOSS>:
  - fixed: no adaptation (Fixed model in the paper).
  - likelihood: increases the likelihood of the generated caption if the listener guesses the correct target (LH model in the paper)
  - rsa_likelihood: generates pragmatic captions incrementally keeping track of the wrong listener guesses and increases the likelihood of the generated caption if the listener guesses the correct target  (RSA LH model in the paper)
  - rsa_likelihood_and_reset: generates pragmatic captions incrementally keeping track of the wrong listener guesses and resets the memory and increases the likelihood of the generated caption if the listener guesses the correct target (RSA LH-Reset model in the paper)
- <SPEAKER_RESET_AFTER>:
  - context: reset after each context
  - domain: reset after each domain
- <LISTENER_ENCODER_PATH>:
  - models/listener/layman_listener_encoder-5-2200.ckpt to use the encoder of the layman listener
  - models/listener/expert_listener_encoder-5-3000.pkl to use the encoder of the expert listener
- <LISTENER_DECODER_PATH>:
  - models/listener/layman_listener_decoder-5-2200.ckpt to use the decoder of the layman listener
  - models/listener/expert_listener_decoder-5-3000.pkl to use the decoder of the expert listener
- <REPORT_PATH>: path of the generated report

### Visualization of speaker adaptation
In order to visualize the speaker adaptation during the different interactions, run:
```
streamlit run visualize_interactions.py
```
The app will run a local HTTP server showing an interface to visualize the reports contained in the folder "data/reports".

### Context generation
In order to generate the features required to generate the contexts, run:
```
python data/preprocess/extract_features.py
```

After having generated the required features, in order to generate the contexts, run:
```
python generate_contexts.py
```

### Listener pre-training
In order to generate the pre-training dataset for the listener starting from the contexts, run:
```
python generate_pretraining_data_from_contexts.py
```

In order to pre-train the listener from the generated pre-training dataset, follow the instructions reported at the following link:
https://github.com/claudiogreco/frontiers2023/tree/main/image_captioning.
