"""
S3 uploader for VMEvalKit - handles both individual files and structured inference outputs.
"""

import os
import boto3
from botocore.config import Config
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime
import hashlib

class S3ImageUploader:
    """Upload images to S3 with public read access."""
    
    def __init__(self, bucket_name: Optional[str] = None):
        """
        Initialize S3 uploader.
        
        Args:
            bucket_name: S3 bucket name (defaults to S3_BUCKET env var)
        """
        self.bucket_name = bucket_name or os.getenv("S3_BUCKET", "vmevalkit")
        # Force us-east-2 region for vmevalkit bucket
        # The bucket is in us-east-2 but AWS_REGION env var might be set to us-east-1
        self.region = "us-east-2"
        
        # Initialize S3 client with signature version 4 and correct region
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
            config=Config(
                region_name=self.region,
                signature_version='s3v4'
            )
        )
        
        # Test prefix for uploaded images
        self.prefix = f"temp_maze_tests/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def upload(self, image_path: str) -> Optional[str]:
        """
        Upload an image to S3 and return a presigned URL for temporary public access.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Presigned URL for temporary access (valid for 1 hour)
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Generate unique key
        file_hash = hashlib.md5(path.name.encode()).hexdigest()[:8]
        key = f"{self.prefix}/{file_hash}_{path.name}"
        
        try:
            # Upload without ACL
            self.s3_client.upload_file(
                str(path),
                self.bucket_name,
                key,
                ExtraArgs={
                    'ContentType': 'image/png'
                }
            )
            
            # Generate presigned URL (valid for 1 hour)
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self.bucket_name,
                    'Key': key
                },
                ExpiresIn=3600  # 1 hour
            )
            
            print(f"[S3] Uploaded {path.name} with presigned URL")
            return url
            
        except Exception as e:
            print(f"[S3] Failed to upload {path.name}: {e}")
            return None
    
    def cleanup(self):
        """Delete all temporary files uploaded in this session."""
        try:
            # List objects with our prefix
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=self.prefix
            )
            
            if 'Contents' in response:
                # Delete all objects
                objects = [{'Key': obj['Key']} for obj in response['Contents']]
                self.s3_client.delete_objects(
                    Bucket=self.bucket_name,
                    Delete={'Objects': objects}
                )
                print(f"[S3] Cleaned up {len(objects)} temporary files")
        except Exception as e:
            print(f"[S3] Cleanup failed: {e}")
    
    def upload_inference_folder(self, inference_dir: str, prefix: Optional[str] = None) -> Dict[str, str]:
        """
        Upload an entire structured inference folder to S3.
        
        This uploads the new structured output format:
        - video/: Generated video files
        - question/: Input images and prompt
        - metadata.json: Inference metadata
        
        Args:
            inference_dir: Path to the inference output folder
            prefix: Optional S3 prefix (defaults to "inferences/{folder_name}")
            
        Returns:
            Dictionary mapping local files to S3 URLs
        """
        inference_path = Path(inference_dir)
        if not inference_path.exists():
            raise FileNotFoundError(f"Inference directory not found: {inference_dir}")
        
        # Default prefix based on folder name
        if not prefix:
            prefix = f"inferences/{inference_path.name}"
        
        uploaded_files = {}
        
        # Walk through all files in the inference directory
        for file_path in inference_path.rglob('*'):
            if file_path.is_file():
                # Create relative path for S3 key
                relative_path = file_path.relative_to(inference_path)
                key = f"{prefix}/{relative_path}"
                
                # Determine content type
                content_type = 'application/octet-stream'
                if file_path.suffix == '.mp4':
                    content_type = 'video/mp4'
                elif file_path.suffix == '.png':
                    content_type = 'image/png'
                elif file_path.suffix == '.json':
                    content_type = 'application/json'
                elif file_path.suffix == '.txt':
                    content_type = 'text/plain'
                
                try:
                    # Upload file
                    self.s3_client.upload_file(
                        str(file_path),
                        self.bucket_name,
                        key,
                        ExtraArgs={'ContentType': content_type}
                    )
                    
                    # Generate presigned URL
                    url = self.s3_client.generate_presigned_url(
                        'get_object',
                        Params={'Bucket': self.bucket_name, 'Key': key},
                        ExpiresIn=86400  # 24 hours for inference results
                    )
                    
                    uploaded_files[str(relative_path)] = url
                    print(f"[S3] Uploaded {relative_path}")
                    
                except Exception as e:
                    print(f"[S3] Failed to upload {relative_path}: {e}")
        
        # Print upload summary
        print(f"\n✅ Inference folder uploaded to S3")
        if any('video/' in path for path in uploaded_files):
            video_urls = [url for path, url in uploaded_files.items() if 'video/' in path]
            print(f"   Videos: {len(video_urls)} file(s) uploaded")
        if any('question/' in path for path in uploaded_files):
            question_files = [path for path in uploaded_files if 'question/' in path]
            print(f"   Question files: {len(question_files)} file(s) uploaded")
        
        return uploaded_files
