#!/usr/bin/env python3
"""Upload rendered ASL videos and manifest to S3/CloudFront.

Usage:
    export AWS_ACCESS_KEY_ID=...
    export AWS_SECRET_ACCESS_KEY=...
    export AWS_REGION=us-east-1
    export S3_BUCKET=my-asl-content-bucket
    export CLOUDFRONT_DOMAIN=d12345.cloudfront.net
    
    uv run python scripts/upload_to_s3.py --input ./output/videos
"""

import argparse
import json
import mimetypes
import os
from datetime import UTC, datetime
from pathlib import Path

import boto3
from botocore.exceptions import ClientError


def upload_file(
    s3_client, 
    file_path: Path, 
    bucket: str, 
    object_name: str | None = None
) -> bool:
    """Upload a file to an S3 bucket."""
    if object_name is None:
        object_name = file_path.name

    try:
        content_type, _ = mimetypes.guess_type(file_path)
        if content_type is None:
            content_type = 'application/octet-stream'
            
        print(f"Uploading {file_path} to s3://{bucket}/{object_name} ({content_type})...")
        
        s3_client.upload_file(
            str(file_path),
            bucket,
            object_name,
            ExtraArgs={'ContentType': content_type}
        )
    except ClientError as e:
        print(f"Error uploading {file_path}: {e}")
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload ASL content to S3 and update manifest"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("./output/videos"),
        help="Directory containing videos and manifest.json"
    )
    parser.add_argument(
        "--bucket", "-b",
        type=str,
        default=os.environ.get("S3_BUCKET"),
        help="S3 bucket name (or set S3_BUCKET env var)"
    )
    parser.add_argument(
        "--region", "-r",
        type=str,
        default=os.environ.get("AWS_REGION", "us-east-1"),
        help="AWS region"
    )
    parser.add_argument(
        "--prefix", "-p",
        type=str,
        default="asl-content/v1",
        help="S3 key prefix/folder"
    )
    
    args = parser.parse_args()
    
    if not args.bucket:
        print("Error: S3 bucket not specified. Use --bucket or set S3_BUCKET env var.")
        return

    s3_client = boto3.client('s3', region_name=args.region)
    cloudfront_domain = os.environ.get("CLOUDFRONT_DOMAIN")
    
    # Load manifest
    manifest_path = args.input / "manifest.json"
    if not manifest_path.exists():
        print(f"Error: Manifest not found at {manifest_path}")
        return
        
    raw_manifest = json.loads(manifest_path.read_text())
    if isinstance(raw_manifest, list):
        manifest_data: dict = {"version": "1.0.0", "items": raw_manifest}
    else:
        manifest_data = raw_manifest

    manifest_version = str(manifest_data.get("version") or "1.0.0")
    items = manifest_data.get("items", [])
    updated_items = []
    
    print(f"Processing {len(items)} items from manifest...")
    
    for item in items:
        if not item.get("id"):
            continue

        # Check if video file exists locally
        local_video_path = Path(
            item.get("localVideoPath")
            or item.get("video_path")
            or item.get("videoPath")
            or ""
        )
        
        # Determine filename for upload
        filename = f"{item['id']}.mp4"  # Default/assumption
        if local_video_path.exists():
            filename = local_video_path.name
        else:
            # Maybe path was absolute or relative differently in manifest vs current execution
            # Try finding it in input dir by ID
            potential_path = args.input / f"{item['id']}.mp4"
            if potential_path.exists():
                local_video_path = potential_path
                filename = potential_path.name
            else:
                print(f"Warning: Video file not found for {item['id']}, skipping upload.")
                updated_items.append(item)
                continue
        
        # Upload video
        s3_key = f"{args.prefix}/{filename}"
        if upload_file(s3_client, local_video_path, args.bucket, s3_key):
            # Update item URL
            if cloudfront_domain:
                url = f"https://{cloudfront_domain}/{s3_key}"
            else:
                url = f"https://{args.bucket}.s3.{args.region}.amazonaws.com/{s3_key}"
                
            # Normalize to the learning app's expected schema.
            item["type"] = item.get("type") or "video"
            item["remoteUrl"] = url
            item["englishText"] = item.get("englishText") or item.get("text") or ""
            item["durationMs"] = item.get("durationMs") or item.get("duration_ms") or 0
            item["gloss"] = item.get("gloss") or []
            item["version"] = item.get("version") or manifest_version
            item["sizeBytes"] = local_video_path.stat().st_size

            # Remove local-only / legacy fields from the published manifest
            for key in (
                "localVideoPath",
                "video_path",
                "video_url",
                "videoPath",
                "text",
                "duration_ms",
                "path",
                "frames",
            ):
                if key in item:
                    del item[key]
                
            updated_items.append(item)
        else:
            print(f"Failed to upload {filename}")
            updated_items.append(item) # Keep original, maybe retry later
            
    # Update manifest content
    manifest_data["items"] = updated_items
    manifest_data["version"] = manifest_version
    manifest_data["updatedAt"] = datetime.now(UTC).isoformat()
    manifest_data["totalItems"] = len(updated_items)
    scenarios = sorted(
        {item.get("scenario") for item in updated_items if item.get("scenario")}
    )
    manifest_data["scenarios"] = scenarios
    
    # Save updated manifest locally first
    updated_manifest_path = args.input / "manifest_public.json"
    updated_manifest_path.write_text(json.dumps(manifest_data, indent=2))
    
    # Upload manifest
    manifest_key = f"{args.prefix}/manifest.json"
    if upload_file(s3_client, updated_manifest_path, args.bucket, manifest_key):
        if cloudfront_domain:
            manifest_url = f"https://{cloudfront_domain}/{manifest_key}"
        else:
            manifest_url = f"https://{args.bucket}.s3.{args.region}.amazonaws.com/{manifest_key}"
            
        print(f"\nSuccess! Manifest uploaded to: {manifest_url}")
        print("Set EXPO_PUBLIC_CONTENT_MANIFEST_URL to this URL in the learning app.")
    else:
        print("Failed to upload manifest.")


if __name__ == "__main__":
    main()
