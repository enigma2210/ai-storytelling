from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.views.decorators.http import require_http_methods
import json
import logging
import threading
import speech_recognition as sr
from .models import Story
from .langchain_services import StorytellingOrchestrator
from .image_composer import ImageComposer

logger = logging.getLogger(__name__)


def index(request):
    """Main page with form and recent stories display."""
    recent_stories = Story.objects.filter(processing_status='completed')[:6]
    return render(request, 'story_generator/index.html', {
        'recent_stories': recent_stories
    })


@csrf_exempt
@require_http_methods(["POST"])
def generate_story(request):
    """
    Handle story generation request.
    Supports both text input and audio file upload with speech-to-text.
    """
    try:
        # Extract form data
        user_prompt = request.POST.get('prompt', '').strip()
        audio_file = request.FILES.get('audio_file')

        # Handle speech-to-text if audio provided
        if audio_file and not user_prompt:
            logger.info("Processing audio file for transcription")
            user_prompt = transcribe_audio(audio_file)
            if not user_prompt:
                return JsonResponse({
                    'error': 'Could not transcribe audio. Please try again or use text input.',
                    'status': 'error'
                }, status=400)

        # Validate input
        if not user_prompt:
            return JsonResponse({
                'error': 'Please provide a text prompt or upload an audio file.',
                'status': 'error'
            }, status=400)

        if len(user_prompt) < 10:
            return JsonResponse({
                'error': 'Prompt must be at least 10 characters long.',
                'status': 'error'
            }, status=400)

        # Create story record
        story = Story.objects.create(
            user_prompt=user_prompt,
            processing_status='pending'
        )

        if audio_file:
            story.audio_file = audio_file
            story.save()

        logger.info(f"Created story record: {story.id}")

        # Start background processing
        threading.Thread(
            target=process_story_generation,
            args=(str(story.id),),
            daemon=True
        ).start()

        return JsonResponse({
            'story_id': str(story.id),
            'status': 'processing',
            'message': 'Story generation started. This may take 1-2 minutes...'
        })

    except Exception as e:
        logger.error(f"Story generation request error: {e}")
        return JsonResponse({
            'error': 'An unexpected error occurred. Please try again.',
            'status': 'error'
        }, status=500)


def process_story_generation(story_id: str):
    """Enhanced story processing with better error handling."""
    try:
        story = Story.objects.get(id=story_id)
        story.processing_status = 'processing'
        story.save()

        orchestrator = StorytellingOrchestrator()

        # Generate text content with validation
        logger.info("Generating story and character description...")
        text_content = orchestrator.generate_story_and_character(story.user_prompt)

        if not text_content.get('story') or not text_content.get('character_description'):
            raise Exception("Failed to generate story content")

        story.story_text = text_content['story']
        story.character_description = text_content['character_description']
        story.save()

        # Generate images with fallback handling
        logger.info("Generating character image...")
        character_img_path = orchestrator.generate_character_image(story.character_description)

        if not character_img_path:
            logger.warning("Character image generation failed, using fallback")
            # Create a simple fallback
            character_img_path = orchestrator._generate_enhanced_placeholder_image(
                story.character_description, 'character'
            )

        logger.info("Generating background image...")
        background_img_path = orchestrator.generate_background_image(story.story_text)

        logger.info("Generating unified scene...")
        unified_img_path = orchestrator.generate_unified_scene(
            story.character_description, story.story_text
        )

        # Save all paths
        story.character_image_path = character_img_path or ''
        story.background_image_path = background_img_path or ''
        story.composed_image_path = unified_img_path or ''
        story.processing_status = 'completed'
        story.save()

        logger.info(f"Story processing completed successfully: {story_id}")

    except Exception as e:
        logger.error(f"Story processing error for {story_id}: {e}")
        try:
            Story.objects.filter(id=story_id).update(processing_status='failed')
        except:
            pass


@require_http_methods(["GET"])
def story_status(request, story_id):
    """
    Check story generation status and return current progress.
    Used for polling updates from the frontend.
    """
    try:
        story = get_object_or_404(Story, id=story_id)

        response_data = {
            'status': story.processing_status,
            'story_id': str(story.id),
            'created_at': story.created_at.isoformat(),
        }

        # Include content if processing is completed
        if story.processing_status == 'completed':
            response_data.update({
                'story_text': story.story_text,
                'character_description': story.character_description,
                'composed_image_url': f"/media/{story.composed_image_path}" if story.composed_image_path else None,
                'character_image_url': f"/media/{story.character_image_path}" if story.character_image_path else None,
                'background_image_url': f"/media/{story.background_image_path}" if story.background_image_path else None,
            })
        elif story.processing_status == 'failed':
            response_data['error'] = 'Story generation failed. Please try again with a different prompt.'

        return JsonResponse(response_data)

    except Exception as e:
        logger.error(f"Status check error: {e}")
        return JsonResponse({
            'error': 'Unable to check status. Please refresh the page.',
            'status': 'error'
        }, status=500)


def transcribe_audio(audio_file):
    """Enhanced audio transcription with better error handling."""
    import os
    import tempfile
    import speech_recognition as sr

    temp_path = None
    wav_path = None

    try:
        from pydub import AudioSegment

        # Get file extension
        file_extension = os.path.splitext(audio_file.name)[1].lower()
        logger.info(f"Processing audio file: {audio_file.name} ({file_extension})")

        # Create temporary file for uploaded audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            for chunk in audio_file.chunks():
                temp_file.write(chunk)
            temp_path = temp_file.name

        logger.info(f"Saved uploaded audio to: {temp_path}")

        # Convert to WAV format
        try:
            audio_segment = AudioSegment.from_file(temp_path)

            # Enhance audio for better recognition
            # Normalize volume
            audio_segment = audio_segment.normalize()

            # Convert to mono if stereo
            if audio_segment.channels > 1:
                audio_segment = audio_segment.set_channels(1)

            # Set sample rate to 16kHz for better recognition
            audio_segment = audio_segment.set_frame_rate(16000)

            # Create WAV file path
            wav_path = temp_path.rsplit('.', 1)[0] + '_enhanced.wav'

            # Export as WAV
            audio_segment.export(wav_path, format="wav")

            logger.info(f"Enhanced audio saved to: {wav_path}")

        except Exception as e:
            logger.error(f"Audio enhancement failed: {e}")
            return None

        # Initialize speech recognizer
        recognizer = sr.Recognizer()

        # Process the enhanced WAV file
        try:
            with sr.AudioFile(wav_path) as source:
                # Adjust for ambient noise
                recognizer.adjust_for_ambient_noise(source, duration=1)
                # Record audio data
                audio_data = recognizer.record(source)

            # Try multiple recognition services for better results
            text = None

            # Try Google Speech Recognition first
            try:
                text = recognizer.recognize_google(audio_data, language='en-US')
                logger.info(f"Google recognition successful: {text[:100]}...")
            except (sr.UnknownValueError, sr.RequestError):
                logger.warning("Google recognition failed, trying alternatives...")

            # If Google fails, try Sphinx (offline) as fallback
            if not text:
                try:
                    text = recognizer.recognize_sphinx(audio_data)
                    logger.info(f"Sphinx recognition successful: {text[:100]}...")
                except:
                    logger.warning("Sphinx recognition also failed")

            return text.strip() if text else None

        except Exception as e:
            logger.error(f"Speech recognition error: {e}")
            return None

    except ImportError:
        logger.error("pydub not installed. Please install: pip install pydub")
        return None
    except Exception as e:
        logger.error(f"Audio transcription error: {e}")
        return None
    finally:
        # Clean up temporary files
        for file_path in [temp_path, wav_path]:
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.info(f"Cleaned up: {file_path}")
                except:
                    pass


@require_http_methods(["GET"])
def story_detail(request, story_id):
    """Display detailed view of a completed story."""
    story = get_object_or_404(Story, id=story_id)
    return render(request, 'story_generator/detail.html', {
        'story': story
    })
