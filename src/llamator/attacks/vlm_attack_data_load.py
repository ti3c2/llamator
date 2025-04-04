import os
import requests


URL = "https://docs.google.com/uc?export=download"


class VLM_ATTACK_TYPES:
    '''Standartized flags to identify type of VLM attack.'''
    LOWRES_DOCS_ATTACK = 0
    M_ATTACK = 1
    TEXT_HALLUCINATION = 2


class GoogleDemoDataDownloader:
    '''Downloads prepared data for VLM attacks'''
    
    def __init__(self, attack_type : VLM_ATTACK_TYPES):
        self.attack_type = attack_type
        
        
    def demo_file_name(self):
        demos = {
            VLM_ATTACK_TYPES.LOWRES_DOCS_ATTACK: "lowres_docs.zip",
            VLM_ATTACK_TYPES.M_ATTACK: "M-Attack-VLM.zip",
            VLM_ATTACK_TYPES.TEXT_HALLUCINATION: "text_image.zip",
        }
        return demos[self.attack_type]


    def demo_folder_id(self):
        demos = {
            VLM_ATTACK_TYPES.LOWRES_DOCS_ATTACK: "1e5w1FPGXuybsxVRLBv-dYi1H9ShshAmB",
            VLM_ATTACK_TYPES.M_ATTACK: "17--NqJtrnpFSe7O-t516opgRC2zzSKyD",
            VLM_ATTACK_TYPES.TEXT_HALLUCINATION: "1PIG3R2aWfvRCngimZ0dzrrUMWn_phJ2q",
        }
        return demos[self.attack_type]
    

    def download_file_from_google_drive(self,
                                        dest_path : str,
                                        file_id: str) -> None:
        '''Downloads file specified in arguments. '''
        session = requests.Session()

        response = session.get(URL, params={'id': file_id}, stream=True)
        token = self._get_confirm_token(response)

        if token:
            params = {'id': file_id, 'confirm': token}
            response = session.get(URL, params=params, stream=True)

        self._save_response_content(response, dest_path)


    def _get_confirm_token(self, response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None


    def _save_response_content(self, response, destination, chunk_size=32768):
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size):
                if chunk:
                    f.write(chunk)
