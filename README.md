# beat-generation-with-mvae

## 목표
- MusicVAE의 논문인 "A hierarchical recurrent variational autoencoder for music"의 내용을 이해하고 논문에서 제안하는 구조의 모델을 구축한다.
- Magenta Project의 MusicVAE와 Groove MIDI Dataset을 활용하여 전처리와 모델 학습을 진행하고 4마디의 드럼 비트가 담긴 MIDI를 생성한다.

## 논문 내용 정리


## 진행
- MusicVAE의 논문에서 제안하는 모델의 컨셉을 파악
  - 저자가 실험에 사용한 Encoder의 구조는 2-layer BiLSTM Encoder이며, 
  - Hierarchical Decoder의 구조는 2-layer LSMTM의 Conductor와 2-layer LSMTM의 Decoder로 구성되어 있는 것을 확인

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
  - logging 및 ckpt save는 100 step 단위를 유지했으며 loss는 0 step에서 399.31445, 100 step에서 158.26624를 기록한 뒤 차차 낮아지는 양상을 보임
  - 훈련은 GPU 사용을 위해 코랩에서 이루어졌으며 제한 시간으로 인해 11200 step까지 진행
  - 11200 step에서 loss는 0.20408921을 기록
  - 훈련을 진행하면서 점차 샘플의 품질이 나아지는 것을 느낄 수 있었으나 최종 스텝의 ckpt로도 장르나 스타일이 뒤섞인 듯한 느낌을 받음
  
- 장르와 스타일을 고려해 데이터를 선택적으로 훈련에 사용해보기로 함 (v2)
  - Groove MIDI Dataset의 파일명에 장르와 BPM, 연주 스타일, 박자의 정보가 존재하여 이를 기준으로 데이터를 편집
  - 장르는 rock, 연주 스타일은 beat로 고정했으며 BPM은 훈련과 샘플링에 영향을 미치지 않는 요소로 판단하여 배제
  - v1과 config는 동일했으며 구글드라이브 용량 문제로 7700 step까지 훈련을 진행
  - 최종 loss는 0.05041269를 기록
  - 4마디로 생성한 8개 샘플 모두 rock beat로 느껴지는 품질 향상을 경험
  - DrumsConverter의 pitch_classes가 디폴트값인 REDUCED_DRUM_PITCH_CLASSES였기에 가상 악기 연주에는 해당 9개 클래스가 모두 있는 707 Core Kit을 사용
  - Crash와 Ride가 많은 샘플과 그렇지 않은 샘플 2개로 32마디 Interpolation을 진행했고 적당한 변주를 포함하여 자연스럽게 transition이 이뤄지는 것을 확인