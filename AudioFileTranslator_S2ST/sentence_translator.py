import requests
import httpx

class SentenceTranslator:
	def __init__(self, src, dst, patience=-1, timeout=60, error_messages_callback=None):
		self.src = src
		self.dst = dst
		self.patience = patience
		self.timeout = timeout
		self.error_messages_callback = error_messages_callback

	def __call__(self, sentence):
		try:
			translated_sentence = []
			# handle the special case: empty string.
			if not sentence:
				return None
			translated_sentence = self.GoogleTranslate(sentence, src=self.src, dst=self.dst, timeout=self.timeout)
			fail_to_translate = translated_sentence[-1] == '\n'
			while fail_to_translate and self.patience:
				translated_sentence = self.GoogleTranslate(translated_sentence, src=self.src, dst=self.dst,
															timeout=self.timeout).text
				if translated_sentence[-1] == '\n':
					if self.patience == -1:
						continue
					self.patience -= 1
				else:
					fail_to_translate = False

			return translated_sentence

		except KeyboardInterrupt:
			if self.error_messages_callback:
				self.error_messages_callback("Cancelling all tasks")
			else:
				print("Cancelling all tasks")
			return

		except Exception as e:
			if self.error_messages_callback:
				self.error_messages_callback(e)
			else:
				print(e)
			return

	def GoogleTranslate(self, text, src, dst, timeout=60, max_retries=5):
		url = 'https://translate.googleapis.com/translate_a/'
		params = 'single?client=gtx&sl=' + src + '&tl=' + dst + '&dt=t&q=' + text
		headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)', 'Referer': 'https://translate.google.com', }

		for retry_count in range(max_retries):
			try:
				response = requests.get(url + params, headers=headers, timeout=timeout)
				if response.status_code == 200:
					response_json = response.json()[0]
					length = len(response_json)
					translation = ""
					for i in range(length):
						translation = translation + response_json[i][0]
					return translation

			except requests.exceptions.ConnectionError:
				with httpx.Client() as client:
					response = client.get(url + params, headers=headers, timeout=timeout)
					if response.status_code == 200:
						response_json = response.json()[0]
						length = len(response_json)
						translation = ""
						for i in range(length):
							translation = translation + response_json[i][0]
						return translation

			except requests.exceptions.Timeout:
				if retry_count < max_retries - 1:
					time.sleep(5)  # Wait for 5 seconds before retrying
					continue

			except KeyboardInterrupt:
				if self.error_messages_callback:
					self.error_messages_callback("Cancelling all tasks")
				else:
					print("Cancelling all tasks")
				return

			except Exception as e:
				if self.error_messages_callback:
					self.error_messages_callback(e)
				else:
					print(e)
				return

		# If all retries fail, return None or handle it as needed
		return None
