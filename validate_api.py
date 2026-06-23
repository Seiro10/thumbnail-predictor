"""
YouTube API Validation Script
Tests API key configuration and diagnoses 403 errors
"""
import os
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_KEY = os.getenv("API_KEY_YOUTUBE")

if not API_KEY or API_KEY == "your_api_key_here":
    print("❌ ERROR: API_KEY_YOUTUBE not set in .env file")
    print("   Please copy .env.template to .env and add your YouTube API key")
    exit(1)

youtube = build("youtube", "v3", developerKey=API_KEY)

# Test channels: mix of working and failing
test_channels = {
    "UCyBpX85hCof66u1rFJYWXIg": "Cozmouz (previously failed)",
    "UC3oVxxmsURlq_UDIjrFckdg": "Cross The Rubicon (previously worked)",
    "UCsEsFyWMp_e6qO-LRjoKnig": "Emeka Moemeka (previously failed)",
    "UCMT1Aw4R4nf_sFNDeuJqc6w": "AI Warehouse (previously worked)",
}

def test_channel_info(channel_id, channel_name):
    """Test 1: Basic channel info (should always work with valid API key)"""
    try:
        response = youtube.channels().list(
            part="snippet,statistics,contentDetails",
            id=channel_id
        ).execute()

        if response['items']:
            item = response['items'][0]
            print(f"  ✅ channels().list() works")
            sub_count = item['statistics'].get('subscriberCount', 'Hidden')
            if sub_count != 'Hidden':
                print(f"     Subscribers: {int(sub_count):,}")
            else:
                print(f"     Subscribers: Hidden")
            upload_playlist = item['contentDetails']['relatedPlaylists']['uploads']
            print(f"     Upload playlist: {upload_playlist}")
            return True, upload_playlist
        else:
            print(f"  ⚠ Channel not found (might be deleted or private)")
            return False, None

    except HttpError as e:
        print(f"  ❌ channels().list() failed")
        print(f"     Error: {e}")
        return False, None

def test_search_api(channel_id, channel_name):
    """Test 2: Search API (where the original error occurred)"""
    try:
        response = youtube.search().list(
            channelId=channel_id,
            part="id,snippet",
            type="video",
            maxResults=5
        ).execute()

        print(f"  ✅ search().list() works")
        print(f"     Found {len(response['items'])} videos")
        return True

    except HttpError as e:
        print(f"  ❌ search().list() failed")
        error_reason = e.error_details[0]['reason'] if e.error_details else "unknown"
        print(f"     Error code: {e.resp.status}")
        print(f"     Reason: {error_reason}")

        if e.resp.status == 403:
            if 'accountDelegationForbidden' in str(e):
                print(f"     This is the Brand Account delegation error!")
                print(f"     Workaround: Use playlistItems API instead")
            elif 'quotaExceeded' in str(e):
                print(f"     Quota exceeded! Check your API quota at:")
                print(f"     https://console.cloud.google.com/apis/api/youtube.googleapis.com/quotas")
            else:
                print(f"     Check API key restrictions at:")
                print(f"     https://console.cloud.google.com/apis/credentials")

        return False

def test_playlist_items_api(upload_playlist_id, channel_name):
    """Test 3: PlaylistItems API (fallback method)"""
    if not upload_playlist_id:
        print(f"  ⏭ Skipping (no upload playlist ID)")
        return False

    try:
        response = youtube.playlistItems().list(
            playlistId=upload_playlist_id,
            part="snippet",
            maxResults=5
        ).execute()

        print(f"  ✅ playlistItems().list() works")
        print(f"     Found {len(response['items'])} videos")
        return True

    except HttpError as e:
        print(f"  ❌ playlistItems().list() failed")
        print(f"     Error: {e}")
        return False

# Run tests
print("=" * 70)
print("YouTube API Validation Report")
print("=" * 70)
print(f"\nAPI Key: {API_KEY[:10]}...{API_KEY[-5:]}\n")

results = {
    'channels_api': [],
    'search_api': [],
    'playlist_items_api': []
}

for channel_id, channel_name in test_channels.items():
    print(f"\n{'─' * 70}")
    print(f"Testing: {channel_name}")
    print(f"Channel ID: {channel_id}")
    print(f"{'─' * 70}")

    # Test 1: Channel info
    print("\n1️⃣  Test channels().list():")
    success, upload_playlist = test_channel_info(channel_id, channel_name)
    results['channels_api'].append(success)

    # Test 2: Search API
    print("\n2️⃣  Test search().list():")
    search_success = test_search_api(channel_id, channel_name)
    results['search_api'].append(search_success)

    # Test 3: PlaylistItems API
    print("\n3️⃣  Test playlistItems().list():")
    playlist_success = test_playlist_items_api(upload_playlist, channel_name)
    results['playlist_items_api'].append(playlist_success)

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

channels_success = sum(results['channels_api'])
search_success = sum(results['search_api'])
playlist_success = sum(results['playlist_items_api'])
total = len(test_channels)

print(f"\n✅ channels().list():      {channels_success}/{total} channels")
print(f"{'✅' if search_success == total else '⚠'} search().list():        {search_success}/{total} channels")
print(f"{'✅' if playlist_success == total else '⚠'} playlistItems().list(): {playlist_success}/{total} channels")

print("\n" + "=" * 70)
print("RECOMMENDATIONS")
print("=" * 70)

if channels_success == total and search_success == total:
    print("\n✅ All tests passed! Your API key is working perfectly.")
    print("   You can use the search().list() method in scraper.py")

elif channels_success == total and playlist_success > search_success:
    print("\n⚠ Search API has issues, but PlaylistItems API works!")
    print("  ")
    print("  ✅ SOLUTION: Use playlistItems().list() instead of search().list()")
    print("     This avoids Brand Account delegation issues.")
    print("  ")
    print("  The improved scraper.py will automatically use this fallback.")

elif channels_success < total:
    print("\n❌ Basic API access is failing. Check your API key:")
    print("  ")
    print("  1. Verify the API key in .env is correct")
    print("  2. Check if YouTube Data API v3 is enabled:")
    print("     https://console.cloud.google.com/apis/library/youtube.googleapis.com")
    print("  ")
    print("  3. Check API key restrictions:")
    print("     https://console.cloud.google.com/apis/credentials")
    print("     - Application restrictions: Should be 'None'")
    print("     - API restrictions: Should include 'YouTube Data API v3'")

else:
    print("\n⚠ Unexpected results. Please check:")
    print("  - Your API quota hasn't been exceeded")
    print("  - No IP or referrer restrictions on the API key")

print("\n" + "=" * 70)


