"""
Script pour re-scraper uniquement Yomi Denzel (vraie chaîne)
et fusionner avec le dataset existant
"""
import os
import requests
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv
from tqdm import tqdm
from datetime import datetime, timezone
import time

load_dotenv()
API_KEY = os.getenv("API_KEY_YOUTUBE")
youtube = build("youtube", "v3", developerKey=API_KEY)

OUTPUT_DIR = "data/thumbnails"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Vraie chaîne Yomi Denzel
YOMI_DENZEL_ID = "UChgE6R4QauGAJAlYiJOcCGw"

def get_channel_videos_via_playlist(channel_id, max_results=300):
    """Get channel videos using playlistItems API"""
    channels_response = youtube.channels().list(
        part="contentDetails",
        id=channel_id
    ).execute()

    if not channels_response['items']:
        return []

    upload_playlist_id = channels_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']

    videos = []
    next_page_token = None

    while len(videos) < max_results:
        response = youtube.playlistItems().list(
            playlistId=upload_playlist_id,
            part="snippet",
            maxResults=min(50, max_results - len(videos)),
            pageToken=next_page_token
        ).execute()

        for item in response["items"]:
            if item["snippet"]["title"] in ["Private video", "Deleted video"]:
                continue

            thumbnails = item["snippet"]["thumbnails"]
            thumb_url = (
                thumbnails.get("maxres", {}).get("url") or
                thumbnails.get("high", {}).get("url") or
                thumbnails.get("medium", {}).get("url")
            )
            videos.append({
                "video_id": item["snippet"]["resourceId"]["videoId"],
                "title": item["snippet"]["title"],
                "published_at": item["snippet"]["publishedAt"],
                "thumbnail_url": thumb_url,
                "channel_id": channel_id,
            })

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

        time.sleep(1)

    return videos

def get_video_stats(video_ids):
    """Get view counts, likes, comments for videos"""
    stats = {}
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i+50]
        response = youtube.videos().list(
            part="statistics,contentDetails",
            id=",".join(batch)
        ).execute()

        for item in response["items"]:
            duration = item["contentDetails"]["duration"]
            stats[item["id"]] = {
                "view_count": int(item["statistics"].get("viewCount", 0)),
                "like_count": int(item["statistics"].get("likeCount", 0)),
                "comment_count": int(item["statistics"].get("commentCount", 0)),
                "duration": duration,
            }
        time.sleep(1)
    return stats

def get_channel_info(channel_id):
    """Get channel subscriber count and name"""
    response = youtube.channels().list(
        part="statistics,snippet",
        id=channel_id
    ).execute()

    if not response["items"]:
        return None

    item = response["items"][0]
    return {
        "subscriber_count": int(item["statistics"].get("subscriberCount", 0)),
        "channel_name": item["snippet"]["title"],
    }

def compute_performance_score(view_count, published_at, subscriber_count):
    """Calculate performance score"""
    published = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
    days_live = max((datetime.now(timezone.utc) - published).days, 1)
    views_per_day = view_count / days_live
    return views_per_day / max(subscriber_count, 1)

def download_thumbnail(video_id, url):
    """Download thumbnail image"""
    if not url:
        return None

    path = f"{OUTPUT_DIR}/{video_id}.jpg"

    if os.path.exists(path):
        return path

    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(path, "wb") as f:
                f.write(response.content)
            return path
    except:
        return None

def is_short_video(duration):
    """Check if video is a YouTube Short (< 60 seconds)"""
    try:
        if 'H' in duration:
            return False
        if 'M' not in duration:
            seconds = int(duration.replace('PT', '').replace('S', ''))
            return seconds < 60
        minutes = int(duration.split('M')[0].replace('PT', ''))
        return minutes == 0
    except:
        return False

# Main
print("🎯 Re-scraping Yomi Denzel (vraie chaîne: 1.48M abonnés)")
print("="*70)

new_data = []

try:
    # Get channel info
    info = get_channel_info(YOMI_DENZEL_ID)
    print(f"\n📺 {info['channel_name']} ({info['subscriber_count']:,} abonnés)")

    # Get videos
    videos = get_channel_videos_via_playlist(YOMI_DENZEL_ID, max_results=300)
    print(f"   Trouvé {len(videos)} vidéos")

    # Get stats
    video_ids = [v["video_id"] for v in videos]
    stats = get_video_stats(video_ids)

    # Process videos
    videos_added = 0
    for video in tqdm(videos, desc="  Traitement"):
        vid_id = video["video_id"]

        if vid_id not in stats:
            continue

        # Filter shorts
        if is_short_video(stats[vid_id].get("duration", "")):
            continue

        view_count = stats[vid_id]["view_count"]

        # Filter < 5000 views
        if view_count < 5000:
            continue

        score = compute_performance_score(
            view_count,
            video["published_at"],
            info["subscriber_count"]
        )

        thumb_path = download_thumbnail(vid_id, video["thumbnail_url"])

        new_data.append({
            **video,
            "view_count": view_count,
            "like_count": stats[vid_id]["like_count"],
            "comment_count": stats[vid_id]["comment_count"],
            "duration": stats[vid_id]["duration"],
            "subscriber_count": info["subscriber_count"],
            "channel_name": info["channel_name"],
            "performance_score": score,
            "thumbnail_path": thumb_path,
            "niche": "Business",
        })
        videos_added += 1

    print(f"\n✅ {videos_added} nouvelles vidéos récupérées")

except Exception as e:
    print(f"❌ Erreur: {e}")

# Load existing dataset and remove old Yomi Denzel entries
print("\n📊 Fusion avec le dataset existant...")
existing_df = pd.read_csv("data/videos.csv")

print(f"   Dataset actuel: {len(existing_df)} vidéos")

# Remove old Yomi Denzel entries (the fake channel)
existing_df_cleaned = existing_df[~existing_df['channel_name'].str.contains('YomiDenzel', case=False, na=False)]
print(f"   Après suppression ancien Yomi: {len(existing_df_cleaned)} vidéos")

# Add new Yomi Denzel data
new_df = pd.DataFrame(new_data)
final_df = pd.concat([existing_df_cleaned, new_df], ignore_index=True)

print(f"   Après ajout nouveau Yomi: {len(final_df)} vidéos")

# Save
final_df.to_csv("data/videos.csv", index=False)

print("\n" + "="*70)
print(f"✅ Dataset mis à jour: {len(final_df)} vidéos au total")
print(f"📁 Fichier: data/videos.csv")
print("\n📊 Nouvelles vidéos Yomi Denzel:")
yomi_new = final_df[final_df['channel_name'] == info['channel_name']]
print(f"   Total: {len(yomi_new)} vidéos")
print(f"   Vues min: {yomi_new['view_count'].min():,}")
print(f"   Vues max: {yomi_new['view_count'].max():,}")
print(f"   Vues moyenne: {yomi_new['view_count'].mean():,.0f}")


