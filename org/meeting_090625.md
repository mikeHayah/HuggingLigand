# Meeting 09.06.2025
Attendees: Ramy Boulos, Michael Hayay, Justus Möller

## Project Name
GitInScene

## Project Tasks (subject to changes)
* extending mood detection to movie genres (Justus)
* translate subtitles (Ramy)
* benchmarking project results (Michael)

## Dataset
Dataset of movie subtitles from the movie "Plan 9 from Outer Space"
https://commons.wikimedia.org/wiki/TimedText:Plan_9_from_Outer_Space_(1959).webm.en.srt

## Preprocessing and Loading the Data (current implementation)
* loading subtitles either via URL with requests module or file locally
* segment into scenes (currently based on time gaps and llm)

## Analysis Workflow
1. Loading Data (already implemented)
2. Divide subtitles into scenes (already implemented)
3.1 Subtitle translation with LLM
3.2 Mood detection with LLM
4. Performance evaluation
5. Visualisation of results

## Software Packages needed
* python-dotenv - loading environment variables
* huggingface_hub - access to huggingface models
* transformers - local LLM deployment
* torch - running local LLM
* requests - loading dataset
* pytest - unit tests

## Lowest Python Version
3.10

## Communication
WhatsApp Group, Video Chat with Google Meet, GitLab Issues
