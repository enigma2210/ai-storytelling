import os
import requests
import base64
import io
import logging
from typing import Dict, Optional, List, Tuple
import google.generativeai as genai
from langchain.prompts import PromptTemplate
from PIL import Image, ImageDraw, ImageFont
import uuid
from django.conf import settings

logger = logging.getLogger(__name__)


class StorytellingOrchestrator:
    """
    Advanced LangChain orchestration with true iterative refinement.
    Implements multi-pass prompt refinement for highest quality outputs.
    """

    def __init__(self):
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
            self.text_model = genai.GenerativeModel('gemini-1.5-flash')

        self.hf_api_key = os.getenv('HUGGINGFACE_API_KEY')
        self.hf_api_url = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"

        # Iterative refinement settings
        self.max_refinement_iterations = 3
        self.quality_threshold = 0.8

        # Conversation memory for context retention
        self.conversation_history = []

    def generate_story_and_character(self, user_prompt: str) -> Dict[str, str]:
        """
        Generate with true iterative refinement using LangChain prompt templates.
        Implements multi-pass quality improvement.
        """
        try:
            logger.info("Starting iterative story generation with refinement")

            # Clear previous conversation
            self.conversation_history = []

            # Phase 1: Initial Generation with Iterative Refinement
            character_name = self._generate_with_refinement(
                "character_name", user_prompt, self._get_name_templates()
            )

            # Phase 2: Story Generation with Context-Aware Refinement
            story = self._generate_with_refinement(
                "story", user_prompt, self._get_story_templates(),
                context={"character_name": character_name}
            )

            # Phase 3: Character Description with Full Context Refinement
            character_description = self._generate_with_refinement(
                "character_description", user_prompt, self._get_character_templates(),
                context={"character_name": character_name, "story": story}
            )

            # Phase 4: Final Quality Assurance Pass
            final_results = self._final_quality_refinement(story, character_description, character_name)

            return final_results

        except Exception as e:
            logger.error(f"Iterative generation error: {e}")
            return self._generate_fallback_content(user_prompt, "Hero")

    def _generate_with_refinement(self, output_type: str, user_prompt: str,
                                  templates: Dict[str, PromptTemplate],
                                  context: Dict = None) -> str:
        """
        Core iterative refinement engine using LangChain templates.
        """
        try:
            if not self.gemini_api_key:
                return self._get_fallback_output(output_type, user_prompt, context)

            context = context or {}

            # Step 1: Initial generation
            initial_template = templates["initial"]
            initial_output = self._generate_from_template(initial_template, user_prompt, context)

            if not initial_output:
                return self._get_fallback_output(output_type, user_prompt, context)

            # Add to conversation history
            self.conversation_history.append({
                "step": f"{output_type}_initial",
                "prompt": user_prompt,
                "output": initial_output,
                "context": context.copy()
            })

            current_output = initial_output

            # Step 2: Iterative refinement loop
            for iteration in range(self.max_refinement_iterations):
                # Assess quality of current output
                quality_score = self._assess_output_quality(current_output, output_type)

                logger.info(f"{output_type} iteration {iteration + 1}, quality: {quality_score}")

                if quality_score >= self.quality_threshold:
                    logger.info(f"{output_type} quality threshold met, stopping refinement")
                    break

                # Generate refinement prompt
                refinement_template = templates["refinement"]
                refinement_context = {
                    **context,
                    "current_output": current_output,
                    "iteration": iteration + 1,
                    "quality_issues": self._identify_quality_issues(current_output, output_type)
                }

                # Refine the output
                refined_output = self._generate_from_template(refinement_template, user_prompt, refinement_context)

                if refined_output and refined_output != current_output:
                    current_output = refined_output

                    # Add refinement to history
                    self.conversation_history.append({
                        "step": f"{output_type}_refinement_{iteration + 1}",
                        "prompt": user_prompt,
                        "output": current_output,
                        "context": refinement_context.copy()
                    })
                else:
                    logger.info(f"{output_type} no improvement in iteration {iteration + 1}")
                    break

            return current_output

        except Exception as e:
            logger.error(f"Refinement generation error for {output_type}: {e}")
            return self._get_fallback_output(output_type, user_prompt, context)

    def _get_name_templates(self) -> Dict[str, PromptTemplate]:
        """Get prompt templates for character name generation with refinement."""
        return {
            "initial": PromptTemplate(
                input_variables=["user_prompt"],
                template="""
                Based on this story prompt: {user_prompt}

                Generate ONE perfect character name for the main protagonist.
                Consider the genre, setting, and cultural context.

                Requirements:
                - Single name or first name + surname only
                - Fits the story's tone and atmosphere
                - Memorable and distinctive
                - Avoid generic names like "John" or "Hero"

                Character Name:
                """
            ),
            "refinement": PromptTemplate(
                input_variables=["user_prompt", "current_output", "quality_issues"],
                template="""
                Original story prompt: {user_prompt}
                Current character name: {current_output}
                Quality issues identified: {quality_issues}

                Improve the character name by addressing the quality issues while maintaining:
                - Relevance to the story context
                - Memorable and pronounceable nature
                - Cultural and genre appropriateness

                If the current name is already excellent, keep it unchanged.

                Improved Character Name:
                """
            )
        }

    def _get_story_templates(self) -> Dict[str, PromptTemplate]:
        """Get prompt templates for story generation with refinement."""
        return {
            "initial": PromptTemplate(
                input_variables=["user_prompt", "character_name"],
                template="""
                Create an engaging story based on: {user_prompt}
                Main character: {character_name}

                Requirements (350-450 words):
                - Use {character_name} consistently throughout
                - Clear story structure: compelling beginning, developed middle, satisfying conclusion
                - Rich sensory details and vivid descriptions
                - Strong character development and growth
                - Appropriate dialogue and action
                - Maintain consistent tone and pacing

                Story:
                """
            ),
            "refinement": PromptTemplate(
                input_variables=["user_prompt", "character_name", "current_output", "quality_issues"],
                template="""
                Original prompt: {user_prompt}
                Character: {character_name}
                Current story: {current_output}
                Issues to address: {quality_issues}

                Improve the story by addressing these specific issues:
                - Enhance character development and consistency
                - Strengthen narrative structure and pacing
                - Add more vivid descriptions and sensory details
                - Improve dialogue and character voice
                - Ensure proper story resolution

                Keep using {character_name} as the protagonist and maintain the core narrative.

                Improved Story:
                """
            )
        }

    def _get_character_templates(self) -> Dict[str, PromptTemplate]:
        """Get prompt templates for character description with refinement."""
        return {
            "initial": PromptTemplate(
                input_variables=["user_prompt", "character_name", "story"],
                template="""
                Create a detailed character description for {character_name}:

                Context:
                - Original prompt: {user_prompt}
                - Story: {story}

                Requirements (200-300 words):
                - Start with "{character_name} is..."
                - Physical appearance: age, height, build, hair, eyes, distinctive features
                - Clothing and accessories appropriate to the story setting
                - Personality traits and demeanor shown in the story
                - Background and role that explains their actions
                - Visual details perfect for illustration

                Character Description:
                """
            ),
            "refinement": PromptTemplate(
                input_variables=["user_prompt", "character_name", "story", "current_output", "quality_issues"],
                template="""
                Context:
                - Prompt: {user_prompt}
                - Character: {character_name}
                - Story: {story}
                - Current description: {current_output}
                - Issues to fix: {quality_issues}

                Improve the character description by:
                - Adding more specific physical details
                - Ensuring consistency with story events and personality
                - Enhancing visual elements for illustration
                - Strengthening the connection to story context
                - Maintaining focus on {character_name}

                Improved Character Description:
                """
            )
        }

    def _generate_from_template(self, template: PromptTemplate, user_prompt: str, context: Dict) -> Optional[str]:
        """Generate output from LangChain template with context."""
        try:
            # Prepare template variables
            template_vars = {"user_prompt": user_prompt, **context}

            # Format the prompt
            formatted_prompt = template.format(**template_vars)

            # Generate with Gemini
            response = self.text_model.generate_content(formatted_prompt)
            return response.text.strip() if response.text else None

        except Exception as e:
            logger.error(f"Template generation error: {e}")
            return None

    def _assess_output_quality(self, output: str, output_type: str) -> float:
        """Assess quality of generated output (0.0 to 1.0)."""
        try:
            quality_score = 0.0

            # Basic length checks
            word_count = len(output.split())
            if output_type == "character_name":
                if 1 <= word_count <= 3:
                    quality_score += 0.4
            elif output_type == "story":
                if 300 <= word_count <= 500:
                    quality_score += 0.3
            elif output_type == "character_description":
                if 150 <= word_count <= 350:
                    quality_score += 0.3

            # Content quality checks
            if output and len(output.strip()) > 0:
                quality_score += 0.2

            # Avoid generic content
            generic_phrases = ['once upon a time', 'the end', 'hero', 'protagonist']
            if not any(phrase in output.lower() for phrase in generic_phrases):
                quality_score += 0.2

            # Check for completeness
            if output_type == "story" and any(word in output.lower() for word in ['concluded', 'finally', 'end']):
                quality_score += 0.2

            return min(quality_score, 1.0)

        except Exception as e:
            logger.error(f"Quality assessment error: {e}")
            return 0.5  # Default moderate quality

    def _identify_quality_issues(self, output: str, output_type: str) -> str:
        """Identify specific quality issues for refinement prompts."""
        issues = []

        word_count = len(output.split())

        if output_type == "character_name":
            if word_count > 3:
                issues.append("name is too long")
            if any(generic in output.lower() for generic in ['hero', 'protagonist', 'character']):
                issues.append("name is too generic")

        elif output_type == "story":
            if word_count < 250:
                issues.append("story is too short, needs more detail")
            if word_count > 500:
                issues.append("story is too long, needs condensing")
            if 'said' not in output.lower() and 'asked' not in output.lower():
                issues.append("needs more dialogue")
            if not any(ending in output.lower() for ending in ['concluded', 'finally', 'end', 'last']):
                issues.append("needs a proper conclusion")

        elif output_type == "character_description":
            if word_count < 150:
                issues.append("description is too brief, needs more physical details")
            if not any(feature in output.lower() for feature in ['hair', 'eyes', 'height', 'clothing']):
                issues.append("missing physical appearance details")

        return "; ".join(issues) if issues else "minor style improvements needed"

    def _final_quality_refinement(self, story: str, character_description: str, character_name: str) -> Dict[str, str]:
        """Final pass to ensure consistency across all outputs."""
        try:
            if not self.gemini_api_key:
                return {"story": story, "character_description": character_description}

            consistency_template = PromptTemplate(
                input_variables=["story", "character_description", "character_name"],
                template="""
                Perform a final consistency check and minor improvements:

                Character: {character_name}
                Story: {story}
                Character Description: {character_description}

                Ensure:
                1. Character name is used consistently
                2. Character traits match between story and description
                3. Physical appearance aligns with story context
                4. No contradictions between outputs

                If changes are needed, provide improved versions. If they're already consistent, return them unchanged.

                Format your response as:
                STORY: [final story]

                CHARACTER: [final character description]
                """
            )

            consistency_prompt = consistency_template.format(
                story=story, character_description=character_description, character_name=character_name
            )

            response = self.text_model.generate_content(consistency_prompt)

            if response.text:
                # Parse the response
                response_text = response.text.strip()
                if "STORY:" in response_text and "CHARACTER:" in response_text:
                    story_part = response_text.split("STORY:")[1].split("CHARACTER:")[0].strip()
                    char_part = response_text.split("CHARACTER:")[1].strip()

                    return {
                        "story": story_part if story_part else story,
                        "character_description": char_part if char_part else character_description
                    }

            # Return original if parsing fails
            return {"story": story, "character_description": character_description}

        except Exception as e:
            logger.error(f"Final refinement error: {e}")
            return {"story": story, "character_description": character_description}

    def _get_fallback_output(self, output_type: str, user_prompt: str, context: Dict = None) -> str:
        """Generate fallback output when API is unavailable."""
        context = context or {}
        character_name = context.get("character_name", "Hero")

        if output_type == "character_name":
            return "Hero"
        elif output_type == "story":
            return f"In a world inspired by '{user_prompt}', {character_name} embarks on an extraordinary adventure filled with challenges and discoveries that test their courage and determination."
        elif output_type == "character_description":
            return f"{character_name} is a brave and determined individual whose distinctive appearance and strong character make them a natural leader in times of crisis."

        return "Generated content"

    # Keep all existing image generation methods unchanged
    def generate_character_image(self, character_description: str) -> Optional[str]:
        """Generate character portrait with fallback handling."""
        try:
            # Validate and enhance character description
            if not character_description or len(character_description.split()) < 5:
                logger.warning("Character description too short, using fallback")
                character_description = "Portrait of a brave, detailed adventurer with distinctive features, realistic digital art style"

            if not self.hf_api_key:
                logger.warning("No HuggingFace API key, using enhanced placeholder")
                return self._generate_enhanced_placeholder_image(character_description, 'character')

            # Enhanced character-only prompt with better keywords
            char_prompt = f"detailed portrait of {character_description[:120]}, character headshot, clean white background, realistic digital art, high quality, professional lighting, no scenery, no landscape"

            headers = {
                "Authorization": f"Bearer {self.hf_api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "inputs": char_prompt,
                "parameters": {
                    "num_inference_steps": 30,  # Increased for better quality
                    "guidance_scale": 8.5,
                    "width": 512,
                    "height": 512
                }
            }

            logger.info(f"Generating character image with prompt: {char_prompt[:100]}...")

            response = requests.post(self.hf_api_url, headers=headers, json=payload, timeout=60)

            if response.status_code == 200:
                logger.info("Character image generated successfully")
                return self._save_image(response.content, 'character')
            else:
                logger.error(f"Character image generation failed: {response.status_code} - {response.text}")
                return self._generate_enhanced_placeholder_image(character_description, 'character')

        except Exception as e:
            logger.error(f"Character image generation error: {e}")
            return self._generate_enhanced_placeholder_image(character_description, 'character')

    def _generate_enhanced_placeholder_image(self, description: str, image_type: str) -> Optional[str]:
        """Generate enhanced placeholder with actual character illustration."""
        try:
            from PIL import Image, ImageDraw, ImageFont
            import random

            # Create a more appealing placeholder
            img = Image.new('RGB', (512, 512), '#2c3e50')  # Dark blue-gray background
            draw = ImageDraw.Draw(img)

            # Create character silhouette
            if image_type == 'character':
                # Draw simple character silhouette
                # Head
                head_color = '#34495e'
                draw.ellipse([180, 120, 332, 272], fill=head_color, outline='#ecf0f1', width=3)

                # Body
                draw.rectangle([210, 260, 302, 400], fill=head_color, outline='#ecf0f1', width=2)

                # Arms
                draw.rectangle([160, 280, 210, 380], fill=head_color, outline='#ecf0f1', width=2)
                draw.rectangle([302, 280, 352, 380], fill=head_color, outline='#ecf0f1', width=2)

                # Simple face features
                # Eyes
                draw.ellipse([200, 160, 220, 180], fill='#ecf0f1')
                draw.ellipse([292, 160, 312, 180], fill='#ecf0f1')

                # Mouth
                draw.arc([230, 200, 282, 230], 0, 180, fill='#ecf0f1', width=3)

            # Add text
            try:
                font = ImageFont.load_default()
            except:
                font = None

            # Title
            title = f"AI Generated {image_type.title()}"
            text_bbox = draw.textbbox((0, 0), title, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            x_pos = (512 - text_width) // 2
            draw.text((x_pos, 50), title, fill='#ecf0f1', font=font)

            # Description preview
            words = description.split()[:8]
            desc_text = ' '.join(words) + "..."
            text_bbox = draw.textbbox((0, 0), desc_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            x_pos = (512 - text_width) // 2
            draw.text((x_pos, 450), desc_text, fill='#bdc3c7', font=font)

            return self._save_image_pil(img, image_type)

        except Exception as e:
            logger.error(f"Enhanced placeholder generation error: {e}")
            return None

    def generate_background_image(self, story_text: str) -> Optional[str]:
        """Generate environment background only."""
        try:
            if not self.hf_api_key:
                return self._generate_placeholder_image(story_text, 'background')

            setting = self._extract_setting(story_text)
            bg_prompt = f"landscape scene {setting}, no people, no characters, empty environment, scenic background, digital art"

            headers = {
                "Authorization": f"Bearer {self.hf_api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "inputs": bg_prompt,
                "parameters": {
                    "num_inference_steps": 25,
                    "guidance_scale": 7.5,
                    "width": 512,
                    "height": 512
                }
            }

            response = requests.post(self.hf_api_url, headers=headers, json=payload, timeout=45)

            if response.status_code == 200:
                return self._save_image(response.content, 'background')
            else:
                return self._generate_placeholder_image(story_text, 'background')

        except Exception as e:
            logger.error(f"Background image error: {e}")
            return self._generate_placeholder_image(story_text, 'background')

    def generate_unified_scene(self, character_description: str, story_text: str) -> Optional[str]:
        """Generate unified scene from combined prompts."""
        try:
            if not self.hf_api_key:
                return self._generate_placeholder_image("Unified scene", 'unified')

            setting = self._extract_setting(story_text)
            unified_prompt = f"full scene illustration: {character_description[:100]} in {setting}, detailed environment, character integrated naturally into scene, professional digital art, high quality"

            headers = {
                "Authorization": f"Bearer {self.hf_api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "inputs": unified_prompt,
                "parameters": {
                    "num_inference_steps": 30,
                    "guidance_scale": 8.0,
                    "width": 768,
                    "height": 512
                }
            }

            response = requests.post(self.hf_api_url, headers=headers, json=payload, timeout=60)

            if response.status_code == 200:
                return self._save_image(response.content, 'unified')
            else:
                return self._generate_placeholder_image("Unified scene", 'unified')

        except Exception as e:
            logger.error(f"Unified scene error: {e}")
            return self._generate_placeholder_image("Unified scene", 'unified')

    def _extract_setting(self, story_text: str) -> str:
        """Extract setting keywords from story."""
        keywords = [
            'forest', 'castle', 'mountain', 'ocean', 'desert', 'city', 'village',
            'cave', 'tower', 'palace', 'garden', 'battlefield', 'ruins', 'temple',
            'bridge', 'meadow', 'river', 'lake', 'hill', 'valley', 'cottage', 'tavern'
        ]

        story_lower = story_text.lower()
        found_settings = [keyword for keyword in keywords if keyword in story_lower]

        return ' '.join(found_settings[:3]) if found_settings else "fantasy landscape"

    def _generate_placeholder_image(self, description: str, image_type: str) -> Optional[str]:
        """Generate placeholder images."""
        try:
            from PIL import Image, ImageDraw, ImageFont

            img = Image.new('RGB', (512, 512))
            draw = ImageDraw.Draw(img)

            colors = {
                'character': [(76, 144, 226), (123, 104, 238)],
                'background': [(50, 205, 50), (34, 139, 34)],
                'unified': [(255, 165, 0), (255, 69, 0)]
            }

            color_start, color_end = colors.get(image_type, colors['unified'])

            for y in range(512):
                ratio = y / 512
                r = int(color_start[0] * (1 - ratio) + color_end[0] * ratio)
                g = int(color_start[1] * (1 - ratio) + color_end[1] * ratio)
                b = int(color_start[2] * (1 - ratio) + color_end[2] * ratio)
                draw.line([(0, y), (512, y)], fill=(r, g, b))

            try:
                font = ImageFont.load_default()
            except:
                font = None

            title = f"AI {image_type.title()}"
            draw.text((20, 20), title, fill='white', font=font)

            words = description.split()[:15]
            y_offset = 60
            for i in range(0, len(words), 6):
                line = ' '.join(words[i:i + 6])
                draw.text((20, y_offset), line, fill='white', font=font)
                y_offset += 30

            return self._save_image_pil(img, image_type)

        except Exception as e:
            logger.error(f"Placeholder generation error: {e}")
            return None

    def _save_image(self, image_data: bytes, prefix: str) -> Optional[str]:
        """Save image data."""
        try:
            filename = f"{prefix}_{uuid.uuid4().hex[:8]}.png"

            if prefix == 'unified':
                subfolder = 'composed_scenes'
            else:
                subfolder = 'generated_images'

            filepath = os.path.join(settings.MEDIA_ROOT, subfolder, filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            with open(filepath, 'wb') as f:
                f.write(image_data)

            return os.path.join(subfolder, filename)

        except Exception as e:
            logger.error(f"Image saving error: {e}")
            return None

    def _save_image_pil(self, pil_image: Image.Image, prefix: str) -> Optional[str]:
        """Save PIL image."""
        try:
            filename = f"{prefix}_{uuid.uuid4().hex[:8]}.png"

            if prefix == 'unified':
                subfolder = 'composed_scenes'
            else:
                subfolder = 'generated_images'

            filepath = os.path.join(settings.MEDIA_ROOT, subfolder, filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            pil_image.save(filepath, 'PNG')
            return os.path.join(subfolder, filename)

        except Exception as e:
            logger.error(f"PIL image saving error: {e}")
            return None

    def _generate_fallback_content(self, user_prompt: str, character_name: str) -> Dict[str, str]:
        """Generate fallback content."""
        story = f"In a world inspired by '{user_prompt}', {character_name} embarks on an extraordinary journey filled with challenges and discoveries."

        character_desc = f"{character_name} is a determined individual whose appearance reflects their adventurous spirit and strong character."

        return {
            'story': story,
            'character_description': character_desc
        }
