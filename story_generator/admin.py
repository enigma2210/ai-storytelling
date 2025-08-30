from django.contrib import admin
from .models import Story


@admin.register(Story)
class StoryAdmin(admin.ModelAdmin):
    list_display = ['user_prompt_short', 'processing_status', 'created_at']
    list_filter = ['processing_status', 'created_at']
    search_fields = ['user_prompt', 'story_text']
    readonly_fields = ['id', 'created_at', 'updated_at']

    def user_prompt_short(self, obj):
        return obj.user_prompt[:100] + "..." if len(obj.user_prompt) > 100 else obj.user_prompt

    user_prompt_short.short_description = "User Prompt"
