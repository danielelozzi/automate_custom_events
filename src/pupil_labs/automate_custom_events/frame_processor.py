from openai import OpenAI
import pandas as pd
import re
import os
import json
import aiohttp
from pupil_labs.automate_custom_events.cloud_interaction import send_event_to_cloud
import asyncio
import logging
import base64
import io
from PIL import Image

logger = logging.getLogger(__name__)

class FrameProcessor:
    def __init__(
        self,
        base64_frames,
        vid_df,
        openai_api_key,
        gemini_api_key,
        model_provider,
        cloud_token,
        recording_id,
        workspace_id,
        prompt_description,
        prompt_codes,
        batch_size,
        start_time_seconds,
        end_time_seconds,
        stop_event=None,
        save_locally=False,
    ):
        # General params
        self.base64_frames = base64_frames
        self.frame_metadata = vid_df
        self.openai_api_key = openai_api_key
        self.gemini_api_key = gemini_api_key
        self.model_provider = model_provider
        self.cloud_token = cloud_token
        self.recording_id = recording_id
        self.workspace_id = workspace_id
        self.batch_size = batch_size
        self.start_time_seconds = int(start_time_seconds)
        self.end_time_seconds = int(end_time_seconds)
        self.stop_event = stop_event
        self.save_locally = save_locally

        if self.model_provider == "OpenAI":
            self.client = OpenAI(api_key=openai_api_key)
        elif self.model_provider == "Local (CLIP)":
            try:
                from transformers import CLIPProcessor, CLIPModel
                import torch
                self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                self.torch = torch
            except ImportError:
                logger.error("Transformers or Torch not installed. Please install them to use CLIP.")
                raise

        self.activities = re.split(r"\s*;\s*", prompt_description)
        self.codes = re.split(r"\s*;\s*", prompt_codes)

        # Initialize activity states
        self.activity_states = {code: False for code in self.codes}

        self.base_prompt = f"""
        You are an experienced video annotator specialized in eye-tracking data analysis.

        **Task:**
        - Analyze the frames of this egocentric video, the red circle in the overlay indicates where the wearer is looking.
        - Identify when any of the specified activities happen in the video based on the visual content (video feed) and the gaze location (red circle).

        **Activities and Corresponding Codes:**

        The activities are:
        {self.activities}

        The corresponding codes are:
        {self.codes}

        **Instructions:**

        - For each frame:
            - Examine the visual elements and the position of the gaze overlay.
            - Determine if any of the specified activities are detected in the frame.
                - If an activity is detected, record the following information:
                    - **Frame Number:** [frame number]
                    - **Timestamp:** [timestamp from the provided dataframe]
                    - **Code:** [corresponding activity code]
                - If an activity is not detected, move to the next frame. 
        - Only consider the activities listed above. Be as precise as possible. 
        - Ensure the output is accurate and formatted correctly.

        **Output Format:**

        ```
        Frame [frame number]: Timestamp - [timestamp], Code - [code]
        ```

        **Examples:**

        - If in frame 25 the user is cutting a red pepper and the timestamp is 65, the output should be:
            ```
            Frame 25: Timestamp - 65, Code - cutting_red_pper
            ```
        - If in frame 50 the user is looking at the rear mirror, the output should be:
            ```
            Frame 50: Timestamp - [timestamp], Code - looking_rear_mirror
            ```
        """

        self.last_event = None

    def is_within_time_range(self, timestamp):
        # Check if the timestamp is within the start_time_seconds and end_time_seconds
        if self.start_time_seconds is not None and timestamp < self.start_time_seconds:
            return False
        if self.end_time_seconds is not None and timestamp > self.end_time_seconds:
            return False
        return True

    async def query_frame(self, index, session):
        # Check if the frame's timestamp is within the specified time range
        timestamp = self.frame_metadata.iloc[index]["timestamp [s]"]
        if not self.is_within_time_range(timestamp):
            # print(f"Timestamp {timestamp} is not within selected timerange")
            return None
        
        if self.model_provider == "Gemini":
            return await self.query_frame_gemini(index, session, timestamp)
        elif self.model_provider == "Local (CLIP)":
            return await self.query_frame_clip(index, timestamp)

        base64_frames_content = [{"image": self.base64_frames[index], "resize": 768}]
        video_gaze_df_content = [self.frame_metadata.iloc[index].to_dict()]

        PROMPT_MESSAGES = [
            {
                "role": "system",
                "content": (self.base_prompt),
            },
            {
                "role": "user",
                "content": f"The frames are extracted from this video and the timestamps and frame numbers are stored in this dataframe: {json.dumps(video_gaze_df_content)}",
            },
            {"role": "user", "content": base64_frames_content},
        ]

        params = {
            "model": "gpt-4o-2024-05-13",  # gpt-4o-2024-05-13 is the old version / gpt-4o-2024-08-06 the newer
            "messages": PROMPT_MESSAGES,
            "max_tokens": 300,
        }
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json",
        }

        retry_count = 0
        max_retries = 5
        while retry_count < max_retries:
            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=params,
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    response_message = result["choices"][0]["message"]["content"]
                    print("Response from OpenAI API:", response_message)

                    # Updated regex pattern to match the new output format
                    pattern = (
                        r"Frame\s(\d+):\sTimestamp\s-\s([\d.]+),\sCode\s-\s(\w+_\w+)"
                    )
                    matches = re.findall(pattern, response_message)

                    if matches:
                        for match in matches:
                            frame_number = int(match[0])
                            timestamp = float(match[1])
                            code = match[2]
                            # # Check if the activity code is valid
                            if code not in self.codes:
                                print("The activity was not detected")
                                continue

                            # Get the current state of the activity
                            activity_active = self.activity_states[code]

                            if not activity_active:
                                # Activity is starting or being detected for the first time
                                self.activity_states[code] = True
                                send_event_to_cloud(
                                    self.workspace_id,
                                    self.recording_id,
                                    code,
                                    timestamp,
                                    self.cloud_token,
                                )
                                logger.info(f"Activity detected: {code}")
                            else:
                                # Activity already detected, ignore
                                logger.debug(
                                    f"Event for {code} already sent - ignoring."
                                )

                        return {
                            "frame_id": frame_number,
                            "timestamp [s]": timestamp,
                            "code": code,
                        }
                    else:
                        print("No match found in the response")
                        return None
                elif response.status == 429:
                    retry_count += 1
                    wait_time = 2**retry_count  # Exponential backoff
                    logger.warning(
                        f"Rate limit hit. Retrying in {wait_time} seconds..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.debug(f"Error: {response.status}")
                    return None
        print("Max retries reached. Exiting.")
        return None

    async def query_frame_gemini(self, index, session, timestamp):
        base64_image = self.base64_frames[index]
        video_gaze_df_content = [self.frame_metadata.iloc[index].to_dict()]
        context_str = f"The frames are extracted from this video and the timestamps and frame numbers are stored in this dataframe: {json.dumps(video_gaze_df_content)}"
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={self.gemini_api_key}"
        
        payload = {
            "contents": [{
                "parts": [
                    {"text": self.base_prompt + "\n\n" + context_str},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": base64_image
                        }
                    }
                ]
            }],
            "generationConfig": {
                "maxOutputTokens": 300
            }
        }

        headers = {"Content-Type": "application/json"}

        retry_count = 0
        max_retries = 6

        while retry_count < max_retries:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    try:
                        response_message = result["candidates"][0]["content"]["parts"][0]["text"]
                        print("Response from Gemini API:", response_message)
                        
                        pattern = r"Frame\s(\d+):\sTimestamp\s-\s([\d.]+),\sCode\s-\s(\w+_\w+)"
                        matches = re.findall(pattern, response_message)
                        
                        if matches:
                            match = matches[0]
                            # frame_number = int(match[0]) 
                            timestamp_res = float(match[1])
                            code = match[2]
                            return self._register_event(code, timestamp_res, index)
                        return None
                    except (KeyError, IndexError) as e:
                        logger.error(f"Error parsing Gemini response: {e}")
                        return None
                elif response.status == 429:
                    retry_count += 1
                    # Exponential backoff: 2, 4, 8, 16, 32, 64 seconds
                    wait_time = 2 ** retry_count
                    logger.warning(f"Gemini Rate limit hit (429). Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Gemini API Error: {response.status} - {await response.text()}")
                    return None
        
        logger.error("Max retries reached for Gemini.")
        return None

    async def query_frame_clip(self, index, timestamp):
        return await asyncio.to_thread(self._run_clip_inference, index, timestamp)

    def _run_clip_inference(self, index, timestamp):
        image_data = base64.b64decode(self.base64_frames[index])
        image = Image.open(io.BytesIO(image_data))

        # Compare image against all activities
        inputs = self.clip_processor(text=self.activities, images=image, return_tensors="pt", padding=True)
        
        with self.torch.no_grad():
            outputs = self.clip_model(**inputs)
        
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        
        # Threshold for detection
        threshold = 0.3 
        max_prob, max_idx = probs.max(dim=1)
        
        if max_prob.item() > threshold:
            code = self.codes[max_idx.item()]
            return self._register_event(code, timestamp, index)
        return None

    def _register_event(self, code, timestamp, frame_number):
        if code not in self.codes:
            return None

        activity_active = self.activity_states[code]
        if not activity_active:
            self.activity_states[code] = True
            if not self.save_locally:
                send_event_to_cloud(
                    self.workspace_id,
                    self.recording_id,
                    code,
                    timestamp,
                    self.cloud_token,
                )
                logger.info(f"Activity detected: {code} (Sent to Cloud)")
            else:
                logger.info(f"Activity detected: {code} (Local Only)")
        else:
            logger.debug(f"Event for {code} already sent - ignoring.")

        return {
            "frame_id": frame_number,
            "timestamp [s]": timestamp,
            "code": code,
        }

    async def binary_search(self, session, start, end, identified_activities):
        if self.stop_event and self.stop_event.is_set():
            return []

        if start >= end:
            return []

        mid = (start + end) // 2

        results = []
        # Process the mid frame and ensure both prompts are evaluated
        mid_frame_result = await self.query_frame(mid, session)
        if mid_frame_result:
            activity = mid_frame_result["code"]
            if activity not in identified_activities:
                identified_activities.add(activity)
                results.append(mid_frame_result)
            left_results = await self.binary_search(
                session, start, mid, identified_activities
            )
            results.extend(left_results)
        else:
            right_results = await self.binary_search(
                session, mid + 1, end, identified_activities
            )
            results.extend(right_results)
        return results

    async def process_batches(self, session, batch_size):
        identified_activities = set()
        all_results = []
        for i in range(0, len(self.base64_frames), batch_size):
            if self.stop_event and self.stop_event.is_set():
                logger.info("Processing stopped by user.")
                break

            end = min(i + batch_size, len(self.base64_frames))
            batch_results = await self.binary_search(
                session, i, end, identified_activities
            )
            all_results.extend(batch_results)
        return all_results

    async def prompting(self, save_path, batch_size):
        async with aiohttp.ClientSession() as session:
            activity_data = await self.process_batches(session, batch_size)
            print("Filtered Activity Data:", activity_data)
            output_df = pd.DataFrame(activity_data)
            output_df.to_csv(
                os.path.join(save_path, "output_detected_events.csv"), index=False
            )
            return output_df
