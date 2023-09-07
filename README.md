https://colab.research.google.com/drive/1LBE283wxR5HDUtRUWsEzQsS-bWY0-2vf#scrollTo=RWk6EKflAvAS

# aicover
초보개발자 ai음성

코렙을 사용하였으므로 코렙을 이용해주세요.

#@title 0. Check GPU
!nvidia-smi

#@title Mount Google Drive
#@title Google 드라이브 마운트
from google.colab import drive
drive.mount('/content/drive')

#@title 1. Install Library for Youtube WAV Download
#@title 1. Youtube WAV 다운로드용 라이브러리 설치
!pip install yt_dlp
!pip install ffmpeg
!mkdir youtubeaudio

#@title Download Youtube WAV
#@title 유튜브 WAV 다운로드
from __future__ import unicode_literals
import yt_dlp
import ffmpeg
import sys

ydl_opts = {
    'format': 'bestaudio/best',
#    'outtmpl': 'output.%(ext)s',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'wav',
    }],
    "outtmpl": 'youtubeaudio/audio',  # this is where you can edit how you'd like the filenames to be formatted
}
def download_from_url(url):
    ydl.download([url])
    # stream = ffmpeg.input('output.m4a')
    # stream = ffmpeg.output(stream, 'output.wav')


with yt_dlp.YoutubeDL(ydl_opts) as ydl:
      url = "https://www.youtube.com/watch?v=7HDeem-JaSY&pp=ygUG7YK57Lm0" #@param {type:"string"}
      download_from_url(url)
      
#@title 2. Install Demucs for Separating Audio
#@title 2. 오디오 분리를 위한 Demucs 설치
!python3 -m pip install -U demucs


#@title Separate Vocal and Instrument/Noise using Demucs
#@title Demucs를 사용하여 보컬과 악기/노이즈 분리
import subprocess
AUDIO_INPUT = "/content/youtubeaudio/audio.wav" #@param {type:"string"}

command = f"demucs --two-stems=vocals {AUDIO_INPUT}"
result = subprocess.run(command.split(), stdout=subprocess.PIPE)
print(result.stdout.decode())


#@title 3. Split The Audio into Smaller Duration Before Training
#@title 3. 훈련 전 오디오를 더 짧은 기간으로 분할
#@markdown don't put space on speaker name
#@markdown 화자 이름에 공백을 넣지 마세요
SPEAKER_NAME = "Hutao" #@param {type:"string"}
!mkdir -p dataset_raw/{SPEAKER_NAME}


from scipy.io import wavfile
import os
import numpy as np
import argparse
from tqdm import tqdm
import json

from datetime import datetime, timedelta

# Utility functions
# 유틸리티 기능

def GetTime(video_seconds):

    if (video_seconds < 0) :
        return 00

    else:
        sec = timedelta(seconds=float(video_seconds))
        d = datetime(1,1,1) + sec

        instant = str(d.hour).zfill(2) + ':' + str(d.minute).zfill(2) + ':' + str(d.second).zfill(2) + str('.001')

        return instant

def GetTotalTime(video_seconds):

    sec = timedelta(seconds=float(video_seconds))
    d = datetime(1,1,1) + sec
    delta = str(d.hour) + ':' + str(d.minute) + ":" + str(d.second)

    return delta

def windows(signal, window_size, step_size):
    if type(window_size) is not int:
        raise AttributeError("Window size must be an integer.")
    if type(step_size) is not int:
        raise AttributeError("Step size must be an integer.")
    for i_start in range(0, len(signal), step_size):
        i_end = i_start + window_size
        if i_end >= len(signal):
            break
        yield signal[i_start:i_end]

def energy(samples):
    return np.sum(np.power(samples, 2.)) / float(len(samples))

def rising_edges(binary_signal):
    previous_value = 0
    index = 0
    for x in binary_signal:
        if x and not previous_value:
            yield index
        previous_value = x
        index += 1

'''
Last Acceptable Values

min_silence_length = 0.3
silence_threshold = 1e-3
step_duration = 0.03/10

'''
# Change the arguments and the input file here
# 여기에서 인수와 입력 파일을 변경하세요.
input_file = "/content/separated/htdemucs/audio/vocals.wav" #@param {type:"string"}
output_dir = f"/content/dataset_raw/{SPEAKER_NAME}"
min_silence_length = 0.6  # The minimum length of silence at which a split may occur [seconds]. Defaults to 3 seconds.
silence_threshold = 1e-4  # The energy level (between 0.0 and 1.0) below which the signal is regarded as silent.
step_duration = 0.03/10   # The amount of time to step forward in the input file after calculating energy. Smaller value = slower, but more accurate silence detection. Larger value = faster, but might miss some split opportunities. Defaults to (min-silence-length / 10.).


input_filename = input_file
window_duration = min_silence_length
if step_duration is None:
    step_duration = window_duration / 10.
else:
    step_duration = step_duration

output_filename_prefix = os.path.splitext(os.path.basename(input_filename))[0]
dry_run = False

print("Splitting {} where energy is below {}% for longer than {}s.".format(
    input_filename,
    silence_threshold * 100.,
    window_duration
    )
)

# Read and split the file
# 파일을 읽고 분할

sample_rate, samples = input_data=wavfile.read(filename=input_filename, mmap=True)

max_amplitude = np.iinfo(samples.dtype).max
print(max_amplitude)

max_energy = energy([max_amplitude])
print(max_energy)

window_size = int(window_duration * sample_rate)
step_size = int(step_duration * sample_rate)

signal_windows = windows(
    signal=samples,
    window_size=window_size,
    step_size=step_size
)

window_energy = (energy(w) / max_energy for w in tqdm(
    signal_windows,
    total=int(len(samples) / float(step_size))
))

window_silence = (e > silence_threshold for e in window_energy)

cut_times = (r * step_duration for r in rising_edges(window_silence))

# This is the step that takes long, since we force the generators to run.
# 발전기를 강제로 실행시키기 때문에 이 단계는 시간이 오래 걸립니다.
print("Finding silences...")
cut_samples = [int(t * sample_rate) for t in cut_times]
cut_samples.append(-1)

cut_ranges = [(i, cut_samples[i], cut_samples[i+1]) for i in range(len(cut_samples) - 1)]

video_sub = {str(i) : [str(GetTime(((cut_samples[i])/sample_rate))),
                       str(GetTime(((cut_samples[i+1])/sample_rate)))]
             for i in range(len(cut_samples) - 1)}

for i, start, stop in tqdm(cut_ranges):
    output_file_path = "{}_{:03d}.wav".format(
        os.path.join(output_dir, output_filename_prefix),
        i
    )
    if not dry_run:
        print("Writing file {}".format(output_file_path))
        wavfile.write(
            filename=output_file_path,
            rate=sample_rate,
            data=samples[start:stop]
        )
    else:
        print("Not writing file {}".format(output_file_path))

with open (output_dir+'\\'+output_filename_prefix+'.json', 'w') as output:
    json.dump(video_sub, output)

#@title 4. Install dependencies for Training
#@제목 4. 훈련을 위한 종속성 설치

#@markdown makesure there are no "so-vits-svc-fork" folder on your Drive before you run this
#@markdown 이 작업을 실행하기 전에 드라이브에 "so-vits-svc-fork" 폴더가 없는지 확인하세요.
!pip install pyworld==0.3.2
!python -m pip install -U pip wheel
%pip install -U ipython
%pip install -U so-vits-svc-fork
!mkdir drive/MyDrive/so-vits-svc-fork


#@title Automatic preprocessing
#@제목 자동 전처리
!svc pre-resample

!svc pre-config

#@title Copy configs file
#@제목 구성 파일 복사
!cp configs/44k/config.json drive/MyDrive/so-vits-svc-fork

F0_METHOD = "dio" #@param ["crepe", "crepe-tiny", "parselmouth", "dio", "harvest"]
!svc pre-hubert -fm {F0_METHOD}

#@title Train
#@title 학습
%load_ext tensorboard
%tensorboard --logdir drive/MyDrive/so-vits-svc-fork/logs/44k
!svc train --model-path drive/MyDrive/so-vits-svc-fork/logs/44k


#@title 5. Inference
#제목 5. 딥러닝 추론


#@markdown remove ".wav" on AUDIO. if you facing "SVC Command not Found", install step 4 depedencies first. Don't put space of folder/file name
from IPython.display import Audio
#@markdown은 오디오에서 ".wav"를 제거합니다. "SVC 명령을 찾을 수 없음"이 발생하는 경우 먼저 4단계 종속성을 설치하십시오. 폴더/파일 이름에 공백을 넣지 마세요.
IPython.display에서 오디오 가져오기

AUDIO = "/content/separated/htdemucs/audio/vocals" #@param {type:"string"}
MODEL = "/content/drive/MyDrive/so-vits-svc-fork/logs/44k/G_199.pth" #@param {type:"string"}
CONFIG = "/content/drive/MyDrive/so-vits-svc-fork/logs/44k/config.json" #@param {type:"string"}
#@markdown Change According to Your Voice Tone. 12 = 1 Octave | -12 = -1 Octave
PITCH = 0 #@param {type:"integer"}

!svc infer {AUDIO}.wav -c {CONFIG} -m {MODEL} -na -t {PITCH}
# Try comment this line below if you got Runtime Error
# 런타임 오류가 발생하면 아래에 이 줄을 주석으로 달아보세요.
try:
  display(Audio(f"{AUDIO}.out.wav", autoplay=True))
except Exception as e:  print("Error:", str(e))


#@title 5.1 Inference Using Pretrained Model
#@제목 5.1 사전 훈련된 모델을 사용한 추론

#@title Credit - https://huggingface.co/spaces/zomehwh/sovits-models (Diganti dengan model Alice karena Model Hololive di Private oleh yang punya)
!mkdir so-vits-test
!wget -N "https://huggingface.co/spaces/zomehwh/sovits-models/resolve/main/models/alice/alice.pth" -P so-vits-test/
!wget -N "https://huggingface.co/spaces/zomehwh/sovits-models/resolve/main/models/alice/config.json" -P so-vits-test/

#@title 5.3 Combine Vocal and Instrument (Song Cover)
#@title 5.3 보컬과 악기 결합(곡 커버)
!pip install pydub
from pydub import AudioSegment

VOCAL = "/content/separated/htdemucs/audio/vocals.out.wav" #@param {type:"string"}
INSTRUMENT = "/content/separated/htdemucs/audio/no_vocals.wav" #@param {type:"string"}

sound1 = AudioSegment.from_file(VOCAL)
sound2 = AudioSegment.from_file(INSTRUMENT)

combined = sound1.overlay(sound2)

combined.export("/content/FinalCover.wav", format='wav')
try:
  display(Audio(f"/content/FinalCover.wav", autoplay=True))
except Exception as e:  print("Error:", str(e))


#@title 6. Additional : Audio Recording
!pip install ffmpeg-python


"""
To write this piece of code I took inspiration/code from a lot of places.
It was late night, so I'm not sure how much I created or just copied o.O
Here are some of the possible references:
https://blog.addpipe.com/recording-audio-in-the-browser-using-pure-html5-and-minimal-javascript/
https://stackoverflow.com/a/18650249
https://hacks.mozilla.org/2014/06/easy-audio-capture-with-the-mediarecorder-api/
https://air.ghost.io/recording-to-an-audio-file-using-html5-and-js/
https://stackoverflow.com/a/49019356
"""
from IPython.display import HTML, Audio
from google.colab.output import eval_js
from base64 import b64decode
import numpy as np
from scipy.io.wavfile import read as wav_read
import io
import ffmpeg

AUDIO_HTML = """
<script>
var my_div = document.createElement("DIV");
var my_p = document.createElement("P");
var my_btn = document.createElement("BUTTON");
var t = document.createTextNode("Press to start recording");

my_btn.appendChild(t);
//my_p.appendChild(my_btn);
my_div.appendChild(my_btn);
document.body.appendChild(my_div);

var base64data = 0;
var reader;
var recorder, gumStream;
var recordButton = my_btn;

var handleSuccess = function(stream) {
  gumStream = stream;
  var options = {
    //bitsPerSecond: 8000, //chrome seems to ignore, always 48k
    mimeType : 'audio/webm;codecs=opus'
    //mimeType : 'audio/webm;codecs=pcm'
  };
  //recorder = new MediaRecorder(stream, options);
  recorder = new MediaRecorder(stream);
  recorder.ondataavailable = function(e) {
    var url = URL.createObjectURL(e.data);
    var preview = document.createElement('audio');
    preview.controls = true;
    preview.src = url;
    document.body.appendChild(preview);

    reader = new FileReader();
    reader.readAsDataURL(e.data);
    reader.onloadend = function() {
      base64data = reader.result;
      //console.log("Inside FileReader:" + base64data);
    }
  };
  recorder.start();
  };

recordButton.innerText = "Recording... press to stop";

navigator.mediaDevices.getUserMedia({audio: true}).then(handleSuccess);


function toggleRecording() {
  if (recorder && recorder.state == "recording") {
      recorder.stop();
      gumStream.getAudioTracks()[0].stop();
      recordButton.innerText = "Saving the recording... pls wait!"
  }
}

// https://stackoverflow.com/a/951057
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

var data = new Promise(resolve=>{
//recordButton.addEventListener("click", toggleRecording);
recordButton.onclick = ()=>{
toggleRecording()

sleep(2000).then(() => {
  // wait 2000ms for the data to be available...
  // ideally this should use something like await...
  //console.log("Inside data:" + base64data)
  resolve(base64data.toString())

});

}
});

</script>
"""

def get_audio():
  display(HTML(AUDIO_HTML))
  data = eval_js("data")
  binary = b64decode(data.split(',')[1])

  process = (ffmpeg
    .input('pipe:0')
    .output('pipe:1', format='wav')
    .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True, quiet=True, overwrite_output=True)
  )
  output, err = process.communicate(input=binary)

  riff_chunk_size = len(output) - 8
  # Break up the chunk size into four bytes, held in b.
  # b에 보관된 청크 크기를 4바이트로 나눕니다.
  q = riff_chunk_size
  b = []
  for i in range(4):
      q, r = divmod(q, 256)
      b.append(r)

  # Replace bytes 4:8 in proc.stdout with the actual size of the RIFF chunk.
  # proc.stdout의 바이트 4:8을 RIFF 청크의 실제 크기로 바꿉니다.
  riff = output[:4] + bytes(b) + output[8:]

  sr, audio = wav_read(io.BytesIO(riff))

  return audio, sr

# 오디오 본인 목소리로 학습시켜서 넣고 싶을 때
  audio, sr = get_audio()

  from scipy.io import wavfile
wavfile.write("my_audio.wav", sr, audio)
