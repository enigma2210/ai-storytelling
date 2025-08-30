from django.db import models
import uuid


class Story(models.Model):
    PROCESSING_STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user_prompt = models.TextField(help_text="User's input prompt for story generation")
    story_text = models.TextField(blank=True, help_text="Generated story content")
    character_description = models.TextField(blank=True, help_text="Generated character description")
    character_image_path = models.CharField(max_length=500, blank=True, help_text="Path to character image")
    background_image_path = models.CharField(max_length=500, blank=True, help_text="Path to background image")
    composed_image_path = models.CharField(max_length=500, blank=True, help_text="Path to final composed image")
    audio_file = models.FileField(upload_to='temp_audio/', blank=True, null=True, help_text="Uploaded audio file")
    processing_status = models.CharField(max_length=20, choices=PROCESSING_STATUS_CHOICES, default='pending')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Generated Story'
        verbose_name_plural = 'Generated Stories'

    def __str__(self):
        return f"Story: {self.user_prompt[:50]}..."
