import itertools
import json
import os
import time

import requests
import grpc
import uuid
import io

from .salute_speech import recognition_pb2
from .salute_speech import recognition_pb2_grpc

from multiprocessing import Event

HOST = "smartspeech.sber.ru"
TIME_DELTA = 60
OAUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"

ENCODINGS_MAP = {
    "pcm": recognition_pb2.RecognitionOptions.PCM_S16LE,
    "opus": recognition_pb2.RecognitionOptions.OPUS,
    "mp3": recognition_pb2.RecognitionOptions.MP3,
    "flac": recognition_pb2.RecognitionOptions.FLAC,
    "alaw": recognition_pb2.RecognitionOptions.ALAW,
    "mulaw": recognition_pb2.RecognitionOptions.MULAW,
}

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_CA_PATH = os.path.join(FILE_DIR, "../certs/russian_trusted_root_ca_pem.crt")


class ASR:

    def __init__(self, config):
        self.token = config.asr.token.get_secret_value()
        self.chunk_size = config.asr.chunk_size
        self.sleep_time = config.asr.sleep_time
        self.sampling_rate = config.asr.sampling_rate
        self.ssl_cred = grpc.ssl_channel_credentials(
            root_certificates=open(ROOT_CA_PATH, "rb").read() if ROOT_CA_PATH else None
        )
        self.refresh_token()
        self.now_listening = Event()
        self.now_listening.clear()

    def __generate_audio_chunks(self, path, chunk_size=None, sleep_time=None):
        if chunk_size is None:
            chunk_size = self.chunk_size
        if sleep_time is None:
            sleep_time = self.sleep_time
        with open(path, 'rb') as f:
            for data in iter(lambda: f.read(chunk_size), b''):
                yield recognition_pb2.RecognitionRequest(audio_chunk=data)
                time.sleep(sleep_time)

    def __generate_audio_chunks_from_bytes(self, audio_data, chunk_size=None, sleep_time=None):
        if chunk_size is None:
            chunk_size = self.chunk_size
        if sleep_time is None:
            sleep_time = self.sleep_time
        audio_io = io.BytesIO(audio_data)
        for data in iter(lambda: audio_io.read(chunk_size), b''):
            yield recognition_pb2.RecognitionRequest(audio_chunk=data)
            time.sleep(sleep_time)

    def __get_token(self):
        headers = {
            "Authorization": "Basic " + self.token,
            "RqUID": str(uuid.uuid4()),
            "Content-Type": "application/x-www-form-urlencoded",
        }
        data = {"scope": "SALUTE_SPEECH_CORP"}
        response = requests.post(OAUTH_URL, headers=headers, data=data, verify=ROOT_CA_PATH)
        data = json.loads(response.text)
        assert "access_token" in data, f"Cant get ASR token. Response: {data}"
        return data["access_token"], data["expires_at"]

    def interrupt(self):
        self.now_listening.clear()

    def refresh_token(self):
        token, expire_deadline = self.__get_token()
        self.token_cred = grpc.access_token_call_credentials(token)
        self.expire_deadline = expire_deadline / 1000

    def recognize(self, audio, mode):
        if time.time() + TIME_DELTA > self.expire_deadline:
            self.refresh_token()
        channel = grpc.secure_channel(
            HOST,
            grpc.composite_channel_credentials(self.ssl_cred, self.token_cred),
        )
        stub = recognition_pb2_grpc.SmartSpeechStub(channel)
        asr_request = recognition_pb2.RecognitionRequest()
        asr_request.options.model = 'general'
        asr_request.options.audio_encoding = ENCODINGS_MAP["pcm"]
        asr_request.options.sample_rate = self.sampling_rate
        asr_request.options.language = "ru-RU"
        asr_request.options.enable_profanity_filter = False
        asr_request.options.enable_multi_utterance = True
        asr_request.options.enable_partial_results = True
        asr_request.options.no_speech_timeout.FromSeconds(2)
        asr_request.options.speaker_separation_options.count = 2
        asr_request.options.speaker_separation_options.enable = True
        self.now_listening.set()
        if mode == "file":
            con = stub.Recognize(
                itertools.chain((asr_request,), self.__generate_audio_chunks(audio))
            )
        else:
            con = stub.Recognize(
                itertools.chain((asr_request,), self.__generate_audio_chunks_from_bytes(audio))
            )

        text = []
        try:
            for resp in con:
                for _, hyp in enumerate(resp.results):
                    if resp.eou:
                        text.append(hyp.normalized_text)

        except grpc.RpcError as err:
            print('RPC error: code = {}, details = {}'.format(err.code(), err.details()))
        except Exception as exc:
            print('Exception:', exc)
        finally:
            full_text = [' '.join(text)]
            channel.close()

        return full_text
