import re
from itertools import accumulate
from typing import TYPE_CHECKING, Dict, List, Optional, Union

from transformers.utils import logging
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput, is_valid_image, load_image
from transformers.processing_utils import Unpack
from transformers.tokenization_utils_base import TextInput, BatchEncoding
# Import the *parent* class
from transformers.models.idefics3.processing_idefics3 import (
    Idefics3Processor,
    Idefics3ProcessorKwargs,
    is_url,
    is_image_or_image_url,
    get_image_prompt_string,
)

if TYPE_CHECKING:
    from transformers.tokenization_utils_base import PreTokenizedInput

logger = logging.get_logger(__name__)


class SmolLMMProcessor(Idefics3Processor):
    """
    A subclass of Idefics3Processor that adds an `allow_mismatch` argument
    to skip the 1:1 match check between #<image> tokens and #images.
    """

    def __call__(
        self,
        images: Union[ImageInput, List[ImageInput], List[List[ImageInput]]] = None,
        text: Union[TextInput, "PreTokenizedInput", List[TextInput], List["PreTokenizedInput"]] = None,
        audio=None,
        videos=None,
        image_seq_len: Optional[int] = None,
        allow_mismatch: bool = False,  # <--- NEW ARG
        **kwargs: Unpack[Idefics3ProcessorKwargs],
    ) -> BatchEncoding:
        """
        Process input for Idefics3. If `allow_mismatch=True`, we skip the error when
        #<image> tokens != #images.

        See `Idefics3Processor.__call__` docstring for details on the other params.
        """
        if text is None and images is None:
            raise ValueError("You must provide either `text` or `images`.")

        # Merge default keyword args for text/images
        output_kwargs = self._merge_kwargs(
            Idefics3ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        # If no override for image_seq_len is passed, use the default self.image_seq_len
        image_seq_len = image_seq_len if image_seq_len is not None else self.image_seq_len

        # Count how many <image> tokens are in each text sample
        n_images_in_text = []
        if text is not None:
            if isinstance(text, str):
                text = [text]
            n_images_in_text = [sample.count(self.image_token.content) for sample in text]

        inputs = BatchFeature()

        # ---------------------------------------------------------------
        # If images are provided, do all the logic that normally raises a mismatch error.
        # We'll skip or warn if allow_mismatch is True.
        # ---------------------------------------------------------------
        if images is not None:
            # Flatten or interpret images
            if is_image_or_image_url(images):
                images = [[images]]
            elif isinstance(images, list) and is_image_or_image_url(images[0]):
                # Original code raises error if mismatch. We'll skip if allow_mismatch.
                if not allow_mismatch and text is not None and sum(n_images_in_text) != len(images):
                    raise ValueError(
                        f"The total number of <image> tokens in the prompts should match "
                        f"the number of images. Found {sum(n_images_in_text)} <image> tokens "
                        f"but {len(images)} images."
                    )
                else:
                    if text is not None and sum(n_images_in_text) != len(images):
                        logger.warning(
                            "Mismatch #<image> tokens vs. #images, but allow_mismatch=True => continuing."
                        )

                # Re-group images to match text samples
                # if text is not None:
                #     cumsum_images_in_text = [0] + list(accumulate(n_images_in_text))
                #     images = [
                #         images[cumsum_images_in_text[i] : cumsum_images_in_text[i + 1]]
                #         for i in range(len(n_images_in_text))
                #     ]
                if text is not None:
                    # Calculate frames per token
                    total_images = len(images)
                    total_tokens = sum(n_images_in_text)
                    if total_images > total_tokens and total_images % total_tokens == 0:
                        frames_per_token = total_images // total_tokens                        
                        # Create new grouping that preserves consecutive frames
                        new_images = []
                        for i in range(len(n_images_in_text)):
                            start_idx = i * frames_per_token * n_images_in_text[i]
                            end_idx = start_idx + (frames_per_token * n_images_in_text[i])
                            new_images.append(images[start_idx:end_idx])
                        images = new_images
                    else:
                        # Original regrouping logic for other cases
                        cumsum_images_in_text = [0] + list(accumulate(n_images_in_text))
                        images = [
                            images[cumsum_images_in_text[i] : cumsum_images_in_text[i + 1]]
                            for i in range(len(n_images_in_text))
                        ]
                else:
                    images = [images]

            elif (
                not isinstance(images, list)
                and not isinstance(images[0], list)
                and not is_image_or_image_url(images[0][0])
            ):
                raise ValueError("Invalid input images. Provide image or list of images or list of list of images.")

            n_images_in_images = [len(sample) for sample in images]
            # Actually load images if they are URLs
            images = [[load_image(im) if is_url(im) else im for im in sample] for sample in images]

            # Let the parent's image_processor handle shape, resizing, etc.
            image_inputs = self.image_processor(images, **output_kwargs["images_kwargs"])
            inputs.update(image_inputs)

            # If we have text, handle expansions
            if text is not None:
                if not allow_mismatch and n_images_in_images != n_images_in_text:
                    raise ValueError(
                        f"Mismatch in #images vs #<image> tokens. We found {n_images_in_text} <image> tokens "
                        f"but have {n_images_in_images} images in each batch."
                    )
                else:
                    if n_images_in_images != n_images_in_text:
                        logger.warning(
                            "Mismatch in #images vs #<image> tokens, but allow_mismatch=True => continuing."
                        )

                # Rows/cols for expanded patch tokens
                image_rows = inputs.pop("rows", [[0] * len(text)])
                image_cols = inputs.pop("cols", [[0] * len(text)])

                fake_image_token = self.fake_image_token.content
                image_token = self.image_token.content
                global_img_token = self.global_image_tag


                prompt_strings = []
                for sample, sample_rows, sample_cols in zip(text, image_rows, image_cols):
                    image_prompt_strings = []
                    for n_rows, n_cols in zip(sample_rows, sample_cols):
                        image_prompt_string = get_image_prompt_string(
                            n_rows,
                            n_cols,
                            image_seq_len,
                            image_token=image_token,
                            fake_token_around_image=fake_image_token,
                            global_img_token=global_img_token,
                        )
                        image_prompt_strings.append(image_prompt_string)

                    split_sample = sample.split(image_token)
                    if len(split_sample) == 0:
                        raise ValueError("Expected <image> token in text, found none.")

                    # Insert expansions for each <image> placeholder
                    combined = split_sample[0]

                    # for i, image_prompt_string in enumerate(image_prompt_strings):
                    #     combined += image_prompt_string + split_sample[i + 1]
                    for i, split_subsample in enumerate(split_sample[1:]):
                        combined += image_prompt_strings[i-1] + split_subsample

                    prompt_strings.append(combined)

                # Now tokenize the text with expansions
                text_inputs = self.tokenizer(text=prompt_strings, **output_kwargs["text_kwargs"])
                inputs.update(text_inputs)

        # -------------------------------------------------------------------
        # If we have text only (no images)
        # -------------------------------------------------------------------
        elif text is not None:
            # no images => zero <image> tokens
            if any(n_images_in_text):
                raise ValueError(
                    f"Found {sum(n_images_in_text)} <image> tokens in text, but no images were passed."
                )
            text_inputs = self.tokenizer(text=text, **output_kwargs["text_kwargs"])
            inputs.update(text_inputs)

        return inputs

    # batch_decode, decode, model_input_names remain the same as parent
    # If you want them identical, no need to override them.