# -*- coding: utf-8 -*-
"""code

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1VHpkLK1s-0yklGypGs89pwuSp4fPUvSS
"""

from transformers import pipeline, logging

# Suppress warnings from transformers
logging.set_verbosity_error()

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
#from transformers import pipeline

#summarizer = pipeline("summarization", model="google/pegasus-xsum")

def summarize_text(text, max_length=130, min_length=30, do_sample=False):
    """
    Summarizes the given text using the pre-trained summarization model.

    Args:
        text: The text to be summarized.
        max_length: The maximum length of the summary.
        min_length: The minimum length of the summary.
        do_sample: Whether to use sampling during generation (can lead to more creative summaries).

    Returns:
        The generated summary.  Returns an error message if summarization fails.
    """
    try:
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=do_sample)[0]['summary_text']
        return summary
    except Exception as e:
        return f"Error during summarization: {e}"

# Example usage:
text_to_summarize = """
Hello, my name is Akash, and I'm a software engineer currently working at HCLTech as a Machine Learning Engineer.
I have a strong background in software development, complemented by my experience in applying machine learning techniques to solve real-world problems within a large organization.
At HCLTech, I've been involved in [mention specific projects or accomplishments, quantifying results whenever possible.
For example: "developing and deploying a machine learning model that improved customer churn prediction by 15%," or "leading a team of engineers in building a new recommendation
system that increased user engagement by 20%"]. My skills encompass a wide range of technologies, including [list key skills: e.g., Python, TensorFlow, PyTorch,
specific cloud platforms like GCP or AWS, specific ML algorithms, big data technologies like Spark or Hadoop]. I'm eager to transition my expertise to Google,
where I believe my skills and experience in AI/ML would be a valuable asset. I'm particularly interested in [mention specific areas of AI/ML that interest you
at Google, aligning with their work if possible]. I'm confident that my dedication to innovation and problem-solving would make me a strong contributor to your team. Thank you for your time and consideration
"""

summary = summarize_text(text_to_summarize)
print(f"Summary:\n{summary}")

!pip install transformers youtube-transcript-api

from transformers import pipeline
from youtube_transcript_api import YouTubeTranscriptApi

def fetch_transcript(video_id):
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    text = " ".join([entry['text'] for entry in transcript])
    return text

def summarize_text(text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    # Split the text into smaller chunks if it's too long
    if len(text) > 1024: # This is for illustration change to an appropriate value
        chunks = [text[i:i + 1024] for i in range(0, len(text), 1024)]
        summary = ""
        for chunk in chunks:
            summary_chunk = summarizer(chunk, max_length=150, min_length=30, do_sample=False)
            summary += summary_chunk[0]['summary_text'] + " "
        return summary.strip()
    else:
        summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
        return summary[0]['summary_text']

def summarize_youtube_video(video_id):
    transcript = fetch_transcript(video_id)
    summary = summarize_text(transcript)
    return summary

video_id = "GWj7-XEp87o"
summary = summarize_youtube_video(video_id)
print(summary)

