"""
Improved YouTube Thumbnail Scraper
Features:
- Dual API support (search + playlistItems fallback)
- Comprehensive error handling
- Rate limiting
- Resume capability
- Detailed logging
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

# ── Configuration ──────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("API_KEY_YOUTUBE")

if not API_KEY or API_KEY == "your_api_key_here":
    print("❌ ERROR: API_KEY_YOUTUBE not set in .env file")
    print("   Please copy .env.template to .env and add your YouTube API key")
    exit(1)

youtube = build("youtube", "v3", developerKey=API_KEY)

# ── Channels cibles (Mixed niches for general analysis) ───
CHANNELS = [
    # AI/Tech niche
    "UCyBpX85hCof66u1rFJYWXIg",  # Cozmouz
    "UCsEsFyWMp_e6qO-LRjoKnig",  # Emeka Moemeka
    "UC3oVxxmsURlq_UDIjrFckdg",  # Cross The Rubicon
    "UCMT1Aw4R4nf_sFNDeuJqc6w",  # AI Warehouse
    "UC-mTYpQ9P9iDDu8H7c-r-SQ",  # Gannon777 / nateherk
    "UCnKyHDGZQxy2n78IS2BkzgQ",  # Dingus Labs
    "UCy9uMtSekidaCDwfnIpzU2g",  # Player2AI
    "UC5sc1ysFs7RfjjEFMuQ3ZQw",  # DougDoug
    "UCTjPBE9BNsmv44wgxWEy2zw",  # Will Kwan
    "UChgE6R4QauGAJAlYiJOcCGw",  # Yomi Denzel (vraie chaîne - 1.48M)

    # Business/Entrepreneurship/Content Creation
    "UCKjyzU_ElNwSXr4b0UzUjdw",  # Philipp Humm
    "UC-PaZZpjgJ61wkK9yKfpe8w",  # Aprilynne Alter
    "UC6kzItRHXCXZyMfOuGiErjQ",  # Ty Myers
    "UCKegeYEy0lJxpkjWRthaRMw",  # Think Media Podcast
    "UCoiGkeGKMrvUdTs0zmFlLnw",  # Gary Ashworth
    "UCxH-b8b2SX4kGSHSVj3NGzA",  # Oussama Ammar
    "UCAuEjBW8MgKN9DyCvjEPFew",  # OttiIie
    "UCpFE8BGxvDhBDYCvD1hpYWw",  # That Nate Black
]

# Channel niche mapping
CHANNEL_NICHES = {
    # AI/Tech
    "UCyBpX85hCof66u1rFJYWXIg": "AI/Tech",
    "UCsEsFyWMp_e6qO-LRjoKnig": "AI/Tech",
    "UC3oVxxmsURlq_UDIjrFckdg": "AI/Tech",
    "UCMT1Aw4R4nf_sFNDeuJqc6w": "AI/Tech",
    "UC-mTYpQ9P9iDDu8H7c-r-SQ": "AI/Tech",
    "UCnKyHDGZQxy2n78IS2BkzgQ": "AI/Tech",
    "UCy9uMtSekidaCDwfnIpzU2g": "AI/Tech",
    "UC5sc1ysFs7RfjjEFMuQ3ZQw": "AI/Tech",
    "UCTjPBE9BNsmv44wgxWEy2zw": "AI/Tech",
    "UChgE6R4QauGAJAlYiJOcCGw": "Business",  # Yomi Denzel (vraie chaîne)

    # Business/Entrepreneurship
    "UCKjyzU_ElNwSXr4b0UzUjdw": "Business",
    "UC-PaZZpjgJ61wkK9yKfpe8w": "Business",
    "UC6kzItRHXCXZyMfOuGiErjQ": "Business",
    "UCKegeYEy0lJxpkjWRthaRMw": "Business",
    "UCoiGkeGKMrvUdTs0zmFlLnw": "Business",
    "UCxH-b8b2SX4kGSHSVj3NGzA": "Business",
    "UCAuEjBW8MgKN9DyCvjEPFew": "Business",
    "UCpFE8BGxvDhBDYCvD1hpYWw": "Business",
}

OUTPUT_DIR = "data/thumbnails"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("data", exist_ok=True)

# Rate limiting settings
REQUEST_DELAY = 1.0  # seconds between API calls

# ── API Methods ────────────────────────────────────────────

def get_channel_videos_via_search(channel_id, max_results=100):
    """
    Get channel videos using search().list() API
    Cost: 100 units per request
    """
    videos = []
    next_page_token = None

    while len(videos) < max_results:
        response = youtube.search().list(
            channelId=channel_id,
            part="id,snippet",
            type="video",
            maxResults=min(50, max_results - len(videos)),
            pageToken=next_page_token
        ).execute()

        for item in response["items"]:
            thumbnails = item["snippet"]["thumbnails"]
            thumb_url = (
                thumbnails.get("maxres", {}).get("url") or
                thumbnails.get("high", {}).get("url") or
                thumbnails.get("medium", {}).get("url")
            )
            videos.append({
                "video_id": item["id"]["videoId"],
                "title": item["snippet"]["title"],
                "published_at": item["snippet"]["publishedAt"],
                "thumbnail_url": thumb_url,
                "channel_id": channel_id,
            })

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

        time.sleep(REQUEST_DELAY)  # Rate limiting

    return videos

def get_channel_videos_via_playlist(channel_id, max_results=100):
    """
    Get channel videos using playlistItems().list() API
    Cost: 1 unit per request (much cheaper!)
    Fallback method for channels with Brand Account restrictions
    """
    # First, get the channel's upload playlist ID
    channels_response = youtube.channels().list(
        part="contentDetails",
        id=channel_id
    ).execute()

    if not channels_response['items']:
        return []

    upload_playlist_id = channels_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']

    # Fetch videos from the upload playlist
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
            # Skip if it's a deleted/private video
            if item["snippet"]["title"] == "Private video" or item["snippet"]["title"] == "Deleted video":
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

        time.sleep(REQUEST_DELAY)  # Rate limiting

    return videos

def get_channel_videos_safe(channel_id, max_results=100):
    """
    Try search API first, fallback to playlistItems if 403 error
    """
    try:
        return get_channel_videos_via_search(channel_id, max_results)
    except HttpError as e:
        if e.resp.status == 403 and 'accountDelegationForbidden' in str(e):
            print(f"  ⚠ Search API blocked (Brand Account), using playlistItems fallback...")
            return get_channel_videos_via_playlist(channel_id, max_results)
        else:
            raise

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
            # Parse duration to filter out shorts
            duration = item["contentDetails"]["duration"]

            stats[item["id"]] = {
                "view_count": int(item["statistics"].get("viewCount", 0)),
                "like_count": int(item["statistics"].get("likeCount", 0)),
                "comment_count": int(item["statistics"].get("commentCount", 0)),
                "duration": duration,
            }

        time.sleep(REQUEST_DELAY)  # Rate limiting

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
    """Calculate performance score (views per day normalized by subscribers)"""
    published = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
    days_live = max((datetime.now(timezone.utc) - published).days, 1)
    views_per_day = view_count / days_live
    return views_per_day / max(subscriber_count, 1)

def download_thumbnail(video_id, url):
    """Download thumbnail image"""
    if not url:
        return None

    path = f"{OUTPUT_DIR}/{video_id}.jpg"

    # Skip if already downloaded
    if os.path.exists(path):
        return path

    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(path, "wb") as f:
                f.write(response.content)
            return path
        else:
            print(f"  ⚠ Failed to download {video_id}: HTTP {response.status_code}")
            return None
    except Exception as e:
        print(f"  ⚠ Error downloading {video_id}: {e}")
        return None

def is_short_video(duration):
    """Check if video is a YouTube Short (< 60 seconds)"""
    # Duration format: PT1M30S (1 minute 30 seconds) or PT45S (45 seconds)
    try:
        if 'H' in duration:  # Has hours, definitely not a short
            return False
        if 'M' not in duration:  # Only seconds
            seconds = int(duration.replace('PT', '').replace('S', ''))
            return seconds < 60
        # Has minutes
        minutes = int(duration.split('M')[0].replace('PT', ''))
        return minutes == 0  # Less than 1 minute
    except:
        return False  # If parsing fails, don't filter it out

# ── Main ───────────────────────────────────────────────────

def main():
    all_data = []
    log_file = open("data/scraper_log.txt", "w", encoding="utf-8")

    log_file.write(f"YouTube Scraper Run - {datetime.now()}\n")
    log_file.write(f"Channels to process: {len(CHANNELS)}\n")
    log_file.write(f"{'='*70}\n\n")

    # Check for existing data to enable resume
    processed_channels = set()
    if os.path.exists("data/videos.csv"):
        try:
            existing_df = pd.read_csv("data/videos.csv")
            processed_channels = set(existing_df['channel_id'].unique())
            print(f"📂 Found existing data: {len(processed_channels)} channels already processed")

            # Ask user if they want to resume or start fresh
            response = input("Resume from existing data? (y/n): ").lower()
            if response == 'y':
                all_data = existing_df.to_dict('records')
                print(f"✅ Resuming with {len(all_data)} existing videos")
            else:
                processed_channels = set()
                all_data = []
                print("🔄 Starting fresh (existing data will be backed up)")
                if os.path.exists("data/videos.csv"):
                    backup_name = f"data/videos_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    os.rename("data/videos.csv", backup_name)
                    print(f"📦 Backed up to {backup_name}")
        except Exception as e:
            print(f"⚠ Could not load existing data: {e}")
            processed_channels = set()

    # Filter out already processed channels
    channels_to_process = [c for c in CHANNELS if c not in processed_channels]

    if not channels_to_process:
        print("✅ All channels already processed!")
        log_file.close()
        return

    print(f"\n🚀 Processing {len(channels_to_process)} channels...\n")

    for channel_id in tqdm(channels_to_process, desc="Channels"):
        try:
            # Get channel info
            info = get_channel_info(channel_id)
            if not info:
                log_msg = f"❌ Channel not found or inaccessible: {channel_id}\n"
                print(f"\n{log_msg.strip()}")
                log_file.write(log_msg)
                continue

            print(f"\n📺 {info['channel_name']} ({info['subscriber_count']:,} subscribers)")
            log_file.write(f"\n📺 {info['channel_name']} (ID: {channel_id})\n")
            log_file.write(f"   Subscribers: {info['subscriber_count']:,}\n")

            # Get videos using safe method (with fallback)
            videos = get_channel_videos_safe(channel_id, max_results=300)

            if not videos:
                log_msg = f"   ⚠ No videos found\n"
                print(log_msg.strip())
                log_file.write(log_msg)
                continue

            print(f"   Found {len(videos)} videos")
            log_file.write(f"   Videos found: {len(videos)}\n")

            # Get video statistics
            video_ids = [v["video_id"] for v in videos]
            stats = get_video_stats(video_ids)

            # Process each video
            videos_added = 0
            for video in tqdm(videos, desc="  Videos", leave=False):
                vid_id = video["video_id"]

                if vid_id not in stats:
                    continue

                # Filter out shorts
                if is_short_video(stats[vid_id].get("duration", "")):
                    continue

                view_count = stats[vid_id]["view_count"]

                # Skip videos with less than 5000 views
                if view_count < 5000:
                    continue

                score = compute_performance_score(
                    view_count,
                    video["published_at"],
                    info["subscriber_count"]
                )

                thumb_path = download_thumbnail(vid_id, video["thumbnail_url"])

                all_data.append({
                    **video,
                    "view_count": view_count,
                    "like_count": stats[vid_id]["like_count"],
                    "comment_count": stats[vid_id]["comment_count"],
                    "duration": stats[vid_id]["duration"],
                    "subscriber_count": info["subscriber_count"],
                    "channel_name": info["channel_name"],
                    "performance_score": score,
                    "thumbnail_path": thumb_path,
                    "niche": CHANNEL_NICHES.get(channel_id, "General"),
                })
                videos_added += 1

            log_file.write(f"   ✅ Added {videos_added} videos (filtered out shorts and 0-view videos)\n")
            print(f"   ✅ Added {videos_added} videos")

            # Save progress after each channel
            df = pd.DataFrame(all_data)
            df.to_csv("data/videos.csv", index=False)

            # Rate limiting between channels
            time.sleep(REQUEST_DELAY)

        except HttpError as e:
            error_msg = f"   ❌ API Error {e.resp.status}: {str(e)[:100]}\n"
            print(error_msg.strip())
            log_file.write(error_msg)

            if e.resp.status == 403:
                log_file.write(f"   ℹ️ Possible causes:\n")
                log_file.write(f"      - API key restrictions\n")
                log_file.write(f"      - Quota exceeded (check: console.cloud.google.com/apis/api/youtube.googleapis.com/quotas)\n")
            elif e.resp.status == 429:
                log_file.write(f"   ⏱ Rate limit hit, waiting 60 seconds...\n")
                time.sleep(60)

            continue

        except Exception as e:
            error_msg = f"   ❌ Unexpected error: {str(e)[:100]}\n"
            print(error_msg.strip())
            log_file.write(error_msg)
            continue

    # ── Final save and summary ─────────────────────────────
    df = pd.DataFrame(all_data)
    df.to_csv("data/videos.csv", index=False)

    log_file.write(f"\n{'='*70}\n")
    log_file.write(f"✅ Scraping complete\n")
    log_file.write(f"Total videos: {len(df)}\n")
    log_file.write(f"Total channels: {df['channel_id'].nunique()}\n")
    log_file.write(f"Thumbnails downloaded: {df['thumbnail_path'].notna().sum()}\n")
    log_file.close()

    print(f"\n{'='*70}")
    print(f"✅ Done — {len(df)} videos scraped from {df['channel_id'].nunique()} channels")
    print(f"📁 Thumbnails: {OUTPUT_DIR}/")
    print(f"📊 Dataset: data/videos.csv")
    print(f"📝 Log: data/scraper_log.txt")

    print(f"\n📈 Top 10 by performance score:")
    print(df.nlargest(10, "performance_score")[["channel_name", "title", "view_count", "performance_score"]])

    print(f"\n📊 Videos by channel:")
    print(df.groupby('channel_name').size().sort_values(ascending=False))

if __name__ == "__main__":
    main()


