#!/usr/bin/env python3
"""
Script to create a new Document AI OCR processor optimized for multi-language support
"""

import os
from google.cloud import documentai
from google.api_core import exceptions

def create_ocr_processor():
    """Create a new OCR processor for multi-language document processing"""
    
    # Configuration
    project_id = "deepam-ai-tutor"
    location = "us"  # Available locations: us, eu
    
    try:
        # Initialize the client
        client = documentai.DocumentProcessorServiceClient()
        parent = client.common_location_path(project_id, location)
        
        # Create processor configuration
        processor = {
            "display_name": "deepam-multilanguage-ocr",
            "type_": "OCR_PROCESSOR",  # Use OCR processor for text extraction
        }
        
        # Create the processor
        print(f"Creating OCR processor in project {project_id}, location {location}...")
        operation = client.create_processor(parent=parent, processor=processor)
        
        # Wait for the operation to complete
        print("Waiting for processor creation to complete...")
        result = operation.result()
        
        # Extract processor details
        processor_id = result.name.split("/")[-1]
        processor_name = result.name
        
        print(f"✅ Successfully created OCR processor!")
        print(f"Processor ID: {processor_id}")
        print(f"Processor Name: {result.display_name}")
        print(f"Processor Type: {result.type_}")
        print(f"Full Resource Name: {processor_name}")
        
        # Update the .env file
        update_env_file(processor_id)
        
        return processor_id
        
    except exceptions.AlreadyExists:
        print("❌ Processor with this name already exists")
        return None
    except Exception as e:
        print(f"❌ Failed to create processor: {e}")
        print("Let's try using the Google Cloud CLI instead...")
        return create_processor_with_gcloud()

def create_processor_with_gcloud():
    """Create processor using gcloud CLI as fallback"""
    try:
        import subprocess
        
        # Create processor using gcloud CLI
        cmd = [
            "gcloud", "ai", "document-ai", "processors", "create",
            "--location=us",
            "--display-name=deepam-multilanguage-ocr",
            "--type=OCR_PROCESSOR",
            "--project=deepam-ai-tutor"
        ]
        
        print("Creating processor using gcloud CLI...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            # Parse the output to get processor ID
            output_lines = result.stdout.strip().split('\n')
            for line in output_lines:
                if 'name:' in line and 'processors/' in line:
                    processor_path = line.split('name:')[1].strip()
                    processor_id = processor_path.split('/')[-1]
                    print(f"✅ Successfully created OCR processor with gcloud!")
                    print(f"Processor ID: {processor_id}")
                    update_env_file(processor_id)
                    return processor_id
        else:
            print(f"❌ gcloud command failed: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"❌ gcloud fallback also failed: {e}")
        return None

def update_env_file(new_processor_id):
    """Update the .env file with the new processor ID"""
    try:
        env_file_path = ".env"
        
        # Read current .env file
        with open(env_file_path, 'r') as f:
            lines = f.readlines()
        
        # Update the processor ID line
        updated_lines = []
        for line in lines:
            if line.startswith("DOC_AI_PROCESSOR_ID="):
                updated_lines.append(f"DOC_AI_PROCESSOR_ID={new_processor_id}\n")
                print(f"📝 Updated .env file with new processor ID: {new_processor_id}")
            else:
                updated_lines.append(line)
        
        # Write back to .env file
        with open(env_file_path, 'w') as f:
            f.writelines(updated_lines)
            
        print("✅ .env file updated successfully")
        
    except Exception as e:
        print(f"⚠️ Failed to update .env file: {e}")
        print(f"Please manually update DOC_AI_PROCESSOR_ID={new_processor_id} in your .env file")

def list_existing_processors():
    """List existing processors to check what's already available"""
    project_id = "deepam-ai-tutor"
    location = "us"
    
    try:
        client = documentai.DocumentProcessorServiceClient()
        parent = client.common_location_path(project_id, location)
        
        print(f"📋 Existing processors in {project_id}:")
        print("-" * 60)
        
        processors = client.list_processors(parent=parent)
        
        ocr_processors = []
        for processor in processors:
            processor_id = processor.name.split("/")[-1]
            print(f"Name: {processor.display_name}")
            print(f"Type: {processor.type_}")
            print(f"ID: {processor_id}")
            print(f"State: {processor.state}")
            print("-" * 60)
            
            if processor.type_ == "OCR_PROCESSOR":
                ocr_processors.append((processor_id, processor.display_name))
        
        if ocr_processors:
            print(f"✅ Found {len(ocr_processors)} existing OCR processor(s):")
            for proc_id, name in ocr_processors:
                print(f"  - {name} (ID: {proc_id})")
            
            choice = input("\nWould you like to use an existing OCR processor? (y/n): ").lower()
            if choice == 'y' and ocr_processors:
                selected_processor = ocr_processors[0][0]  # Use the first OCR processor
                print(f"Using existing processor: {selected_processor}")
                update_env_file(selected_processor)
                return selected_processor
        
        return None
        
    except Exception as e:
        print(f"❌ Failed to list processors: {e}")
        return None

if __name__ == "__main__":
    print("🔧 Document AI OCR Processor Setup")
    print("=" * 50)
    
    # First, check if there are existing OCR processors
    existing_processor = list_existing_processors()
    
    if not existing_processor:
        print("\n🆕 Creating new OCR processor...")
        new_processor = create_ocr_processor()
        
        if new_processor:
            print(f"\n🎉 Setup complete! New processor ID: {new_processor}")
            print("\n📋 Next steps:")
            print("1. Restart your backend server to use the new processor")
            print("2. Test with a Kannada document")
            print("3. The new processor should handle multi-language text much better")
        else:
            print("\n❌ Failed to create processor. Please check your GCP permissions.")
    else:
        print(f"\n🎉 Setup complete! Using existing processor: {existing_processor}")
        print("\n📋 Next steps:")
        print("1. Restart your backend server to use the updated processor")
        print("2. Test with a Kannada document")
