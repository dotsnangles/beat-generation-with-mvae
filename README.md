# beat-generation-with-mvae
- results 폴더에 생성한 MIDI 파일들이 모여 있습니다. v1부터 v3까지가 개발한 모델로 생성한 파일들입니다.
- training log는 save_model 폴더 안에 버전별로 training_log.txt의 이름으로 들어가 있습니다.
- data_selection.ipynb 외에는 모두 Magenta 라이브러리의 스크립트 실행을 위한 파일들입니다.
- magenta/magenta/models/music_vae/configs.py 안의 hierdec-drum_4bar, hierdec-drum_4bar_v2, hierdec-drum_4bar-no_ar가 제가 작성한 config들이며 실제로 사용된 것은 hierdec-drum_4bar, hierdec-drum_4bar_v2 두 가지입니다.
- config의 내용과 더불어 개발 관련 사항은 개발 진행 순서에 최대한 자세히 정리해두었습니다.
- 모델의 ckpt는 깃헙의 용량 제한 문제로 업로드하지 않았습니다. 필요하신 경우 말씀해주시면 보내드리도록 하겠습니다.
- 개발 과정 중간에 ckpt를 .gitignore에 포함하지 않고 커밋을 해 push가 안 되는 상황이 발생해 시간 관계상 git을 새로 시작했습니다. 백업을 해둔 버전이 있으므로 필요하신 경우 보내드리도록 하겠습니다.

## 목차
- 목표
- 개념 정리
- 개발 진행 순서

## 목표
- MusicVAE의 논문인 "A hierarchical recurrent variational autoencoder for music"의 내용을 이해하고 논문에서 제안하는 구조의 모델을 구축합니다.
- Magenta Project의 MusicVAE와 Groove MIDI Dataset을 활용하여 전처리와 모델 학습을 진행하고 4마디의 드럼 비트가 담긴 MIDI를 생성합니다.

## 개념 정리
Autoencoder와 VAE의 가장 큰 차이는 Encoder의 출력이 single point인 latent representation인지 아니면 latent distribution인지에 있습니다. VAE의 Encoder는 확률 밀도 함수를 학습한 모델이라고도 볼 수 있을 것입니다. 실제로 정규분포의 평균과 표준편차가 코드에 반영되어 있는 것을 확인할 수 있으며, 생성시 영향을 미치는 Temperature 역시 출력 분포에 영향을 주기 위한 장치라고 이해할 수 있겠습니다.  

VAE는 Encoder의 출력인 확률 분포에서 Sampling을 수행하고 다시 Decoder를 통해 reconstruction을 진행하는 구조로 설계되었으며, 생성 모델로서 VAE가 좋은 성능을 발휘하기 위해서는 분포의 continuity와 completeness의 regularity를 확보해야 합니다. 이는 latent space에서 sampling한 point가 서로 가까울 수록 비슷하게 복원되어야 하고 복원된 모습은 목표로한 도메인에 적합한 의미를 가져야 한다는 뜻입니다.

손실함수는 reconstruction과 regularisation 두 가지에 대한 것이며 전자는 encoding과 decoding을 효율적으로 수행하기 위한 것이고 후자는 latent space의 regularity를 높히기 위한 것입니다. 특히 후자는 출력 분포를 표준 정규 분포에 가깝게 만들기 위한 것으로 출력 분포와 표준 정규 분포 사이의 Kulback-Leibler divergence를 통해 표현됩니다. 가중치 업데이트는 다른 딥러닝 모델과 동일하게 역전파 기법을 사용합니다.

"A hierarchical recurrent variational autoencoder for music"에서 제안한 개선 사항은 기존의 RNN Decoder에 conductor를 추가하는 것입니다. conductor가 먼저 latent code를 입력으로 받아 각 마디별로 embedding을 생성한 뒤 RNN Decoder가 각각의 embedding으로부터 마디를 생성하게 됩니다. 보다 나은 representation learning의 구조이기 때문에 기존의 RNN VAE가 가지고 있었던 sequence가 길어질 수록 성능이 떨어졌던 문제를 다소 극복할 수 있었습니다.

MusicVAE를 통해 VAE를 처음 접했기에 사전 지식이 부족한 상태에서 VAE라는 구조의 기저에 어떤 개념들이 깔려 있는지를 파악하기 위해 다양한 자료를 탐색했습니다. "A hierarchical recurrent variational autoencoder for music"의 내용을 살펴본 뒤 추가로 참고하기 위해 논문들을 몇편 정리해두었으나 아직 그 내용을 모두 살펴보지는 못 했습니다.

차원 축소와 Autoencoder부터 개념 정리를 해둔 [Understanding Variational Autoencoders (VAEs)](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73)의 내용이 지금 저에게 가장 많은 통찰을 줄 것이라 판단하여 해당 문서를 중심으로 내용을 정리했으며, 금일 이후에 수학적 내용에 대해서 더 자세히 살펴볼 예정입니다.

다음의 내용은 해당 문서의 개념 소개 중 주요한 내용을 정리한 것입니다.

### Two Most Important Notations
- random variable z
- p(z) the distribution (or the density, depending on the context) of this random variable (z)

### Autoencoder
- dimensionality reduction and autoencoder
- both ideas are related to each others
- autoencoders cannot be used to generate new data

#### Encoder and Decoder
- encoder: the process that produce the “new features” representation from the “old features” representation (by selection or by extraction)
- decoder: the reverse process
- encoder: compress the data (from the initial space to the encoded space, also called latent space)
- decoder: decompress them

#### compression can be lossy, meaning that a part of the information is 
- lost during the encoding process and cannot be recovered when decoding
- among possible encoders and decoders, we are looking for the pair that keeps 
- the maximum of information when encoding and, so, has 
- the minimum of reconstruction error when decoding

#### General Idea of Autoencoders
- setting an encoder and a decoder as neural networks and 
- to learn the best encoding-decoding scheme 
- using an iterative optimisation process
- compare the encoded-decoded output with the initial data and 
- backpropagate the error through the architecture to update the weights of the networks
- autoencoder architecture (encoder+decoder) creates a bottleneck for data 
- that ensures only the main structured part of the information can go through and be reconstructed

### Variational Autoencoder
- Variational Autoencoders that are regularised versions of autoencoders making the generative process possible
- mathematical presentation of VAEs based on variational inference(?)
- VAE is an autoencoder whose encodings distribution is regularised during the training 
- in order to ensure that its latent space has good properties allowing us to generate some new data. 
- Moreover, the term “variational” comes from the close relation there is 
- between the regularisation and the variational inference method in statistics.

#### a variational autoencoder can be defined as being an autoencoder 
- whose training is regularised to avoid overfitting and 
- ensure that the latent space has good properties that enable generative process

#### as a standard autoencoder, a variational autoencoder is an architecture 
- composed of both an encoder and a decoder and 
- that is trained to minimise the reconstruction error 
- between the encoded-decoded data and the initial data.

#### in order to introduce some regularisation of the latent space, 
- we proceed to a slight modification of the encoding-decoding process: 
- instead of encoding an input as a single point, 
- we encode it as a distribution over the latent space.
  
#### VAE is trained as follows:
- the input is encoded as distribution over the latent space
- a point from the latent space is sampled from that distribution
- the sampled point is decoded and the reconstruction error can be computed
- the reconstruction error is backpropagated through the network

#### Difference between autoencoder (deterministic) and variational autoencoder (probabilistic)
![Difference between autoencoder (deterministic) and variational autoencoder (probabilistic)](https://miro.medium.com/max/1400/1*ejNnusxYrn1NRDZf4Kg2lw@2x.png)  


#### the loss function that is minimised when training a VAE is composed of 
- a “reconstruction term” (on the final layer), that tends to make the encoding-decoding scheme as performant as possible, and 
- a “regularisation term” (on the latent layer), that tends to regularise the organisation of the latent space 
- by making the distributions returned by the encoder close to a standard normal distribution.
- That regularisation term is expressed as the Kulback-Leibler divergence between the returned distribution and a standard Gaussian

#### In variational autoencoders, the loss function is composed of 
- a reconstruction term (that makes the encoding-decoding scheme efficient) and 
- a regularisation term (that makes the latent space regular)

![In variational autoencoders, the loss function is composed of a reconstruction term (that makes the encoding-decoding scheme efficient) and a regularisation term (that makes the latent space regular)](https://miro.medium.com/max/1400/1*Q5dogodt3wzKKktE0v3dMQ@2x.png)

#### The regularity that is expected from the latent space 
- in order to make generative process possible can be expressed through two main properties: 
- continuity (two close points in the latent space should not give two completely different contents once decoded) and 
- completeness (for a chosen distribution, a point sampled from the latent space should give “meaningful” content once decoded)

#### we have to regularise both the covariance matrix and the mean of the distributions returned by the encoder. 
- In practice, this regularisation is done by 
- enforcing distributions to be close to a standard normal distribution (centred and reduced). 
- This way, we require 
- the covariance matrices to be close to the identity, preventing punctual distributions, and 
- the mean to be close to 0, preventing encoded distributions to be too far apart from each others.

#### The returned distributions of VAEs have to be regularised to obtain a latent space with good properties
![The returned distributions of VAEs have to be regularised to obtain a latent space with good properties](https://miro.medium.com/max/1400/1*9ouOKh2w-b3NNOVx4Mw9bg@2x.png)

#### Regularisation tends to create a “gradient” over the information encoded in the latent space
![Regularisation tends to create a “gradient” over the information encoded in the latent space](https://miro.medium.com/max/1400/1*79AzftDm7WcQ9OfRH5Y-6g@2x.png)


### Hierarchical Decoder
#### Rather than using the latent code to initialize the note RNN decoder directly, 
- first pass the code to a “conductor” RNN that outputs a new embedding for each bar of the output. 
- The note RNN then generates each of the 16 bars independently, 
- conditioned on the embeddings instead of the latent code itself. 
- then sample autoregressively from the note decoder.

#### this conditional independence is found to be an important feature of our architecture. 
- Since the model could not simply fall back on autoregression in the note decoder to optimize the loss during training, 
- it gained a stronger reliance on the latent code to reconstruct the sequences.

## 개발 진행 순서
- MusicVAE의 논문에서 제안하는 모델의 컨셉을 파악
  - 저자가 실험에 사용한 Encoder의 구조는 2-layer BiLSTM Encoder이며, 
  - Hierarchical Decoder의 구조는 2-layer LSTM의 Conductor와 2-layer LSTM의 Decoder로 구성되어 있는 것을 확인

- 과제의 우선 순위를 MIDI 생성에 두고 Magenta Project내 MusicVAE의 구성 요소를 탐색
  - 데이터 전처리부터 모델 정의, 학습, 생성에 이르기까지 모든 작업과 관련된 코드가 준비가 되어 있는 것을 파악
  - 편집 모드로 Magenta 라이브러리를 설치한 후 개발을 시작

- 출력물을 미리 확인하기 위해 사전학습된 cat-drums_2bar_small.lokl를 다운로드하여 샘플링을 진행
  - 2마디의 드럼 비트가 생성되는 것을 확인했으며 Ableton Live를 사용해 가상악기를 입혀 재생해봄

- Magenta 라이브러리에 준비된 convert_dir_to_note_sequences 스크립트를 활용해 Groove MIDI Dataset을 tfrecord 형식으로 변환
  - music_vae 폴더에 큰 데이터를 sharding할 수 있게 해주는 preprocess_tfrecord가 존재했으나 모든 데이터를 활용할 계획이었기 때문에 사용하지 않음
  - 추후 서술할 config 작성에서 언급하는 DrumsConverter의 설정에 따라 tfrecord를 미리 전처리하는 내용이었으나 불필요하다고 판단

- configs에서 hierdec-mel_16bar의 구성이 BidirectionalLstmEncoder와 HierarchicalLstmDecoder로 이루어져 있으며 단일 채널인 것을 확인
  - 해당 config를 수정하여 hierdec-drum_4bar의 config를 작성
  - cat-drums_2bar_small의 config를 참조하여 batch_size, max_seq_len, enc_rnn_size, dec_rnn_size, free_bits, max_beta, sampling_schedule, sampling_rate를 설정
    - max_seq_len은 4마디의 드럼 비트 생성을 위해 DrumsConverter의 steps_per_quarter, quarters_per_bar, slice_bars를 각각 4, 4, 4로 설정한 뒤 (1마디 당 16비트로 4마디 기준 총 64비트 구성) 그에 맞춰 64로 설정
    - MusicVAE의 README에 free_bits를 높히고 max_beta를 낮추는 것이 KL loss의 효과를 낮춰 더 나은 reconstruction과 더 못 한 random sample 결과로 이어진다는 내용이 있었음
      - 한정된 시간 동안 진행된 프로젝트인 만큼 더 나은 random sampling에 초점을 맞춘 cat-drums_2bar_small.lokl의 설정을 그대로 유지함
  - HierarchicalLstmDecoder의 level_lengths는 max_seq_len에 호응하도록 8, 8로 설정했으며 autoregression을 위해 disable_autoregression을 False로 변경
  - z_size는 참고한 config인 hierdec-mel_16bar의 값을 유지
  - sampling_schedule은 논문의 내용과 동일한 inverse_sigmoid로 설정했으며 sampling_rate는 다른 드럼 config들에서 반복적으로 사용된 1000으로 적용
  - 전체적으로 논문에서 제안한 모델의 구조는 그대로 가지고 오되 드럼 비트 랜덤 샘플링에 적합하다고 판단한 config를 작성

- Groove MIDI Dataset 전체를 tfrecord로 변환한 뒤 v1 훈련을 진행
  - VAE 훈련 경험이 없었기 때문에 디폴트값인 200000 steps로 훈련을 진행하며 loss를 관찰하고 ckpt마다 MIDI 생성을 시도해보기로 함
  - learning_rate, decay_rate, min_learning_rate 역시 마찬가지로 디폴트값을 적용하여 진행 (각각 0.001, 0.9999, 0.00001)
  - logging 및 ckpt save는 100 step 단위를 유지했으며 loss가 0 step에서 399.31445, 100 step에서 158.26624를 기록한 뒤 차차 낮아지는 양상을 보이는 것을 관찰
  - 훈련은 GPU 사용을 위해 코랩에서 이루어졌으며 제한 시간으로 인해 11200 step까지 진행
  - 11200 step에서 loss는 0.20408921을 기록
  - 훈련을 진행하면서 점차 샘플의 품질이 나아지는 것을 느낄 수 있었으나 최종 스텝의 ckpt로도 장르나 스타일이 뒤섞인 듯한 느낌을 받음
  
- 장르와 스타일을 고려해 데이터를 선택적으로 훈련에 사용해보기로 함 (v2)
  - Groove MIDI Dataset의 파일명에 장르와 BPM, 연주 스타일, 박자의 정보가 존재하여 이를 기준으로 데이터를 편집
  - 장르는 rock, 연주 스타일은 beat로 고정했으며 BPM은 훈련과 샘플링에 영향을 미치지 않는 요소로 판단하여 선택 기준에서 배제
  - v1과 config는 동일했으며 구글드라이브 용량 문제로 7700 step까지 훈련을 진행
  - 최종 loss는 0.05041269를 기록
  - 4마디로 생성한 8개 샘플 모두 rock beat로 느껴지는 품질 향상을 경험
  - DrumsConverter의 pitch_classes가 디폴트값인 REDUCED_DRUM_PITCH_CLASSES였기에 가상 악기 연주에는 해당 9개 클래스가 모두 있는 707 Core Kit을 사용
  - Crash와 Ride가 많은 샘플과 그렇지 않은 샘플 2개로 32마디 Interpolation을 진행했고 적당한 변주를 포함하여 자연스럽게 transition이 이뤄지는 것을 확인

- 개념에 대해 추가 학습을 진행하며 latent vector에 대해 좀 더 이해하게 되었고 z_size를 512에서 256으로 조정한 뒤 v3 학습 진행 (v3)
  - v2와 동일한 데이터셋을 사용
  - 구글드라이브 용량 문제로 6600 step까지 훈련을 진행 
  - 최종 loss는 0.07263463을 기록
  - 시간 관계상 8개의 MIDI를 샘플링한 후 청취해봤으며 interpolation은 생략함
