import typing
from typing import List, Optional, Tuple, Union

import sounddevice as sd
from loguru import logger

from vocode.streaming.input_device.microphone_input import (
    MicrophoneInput as StreamingMicrophoneInput,
)
from vocode.streaming.output_device.blocking_speaker_output import (
    BlockingSpeakerOutput as BlockingStreamingSpeakerOutput,
)
from vocode.turn_based.input_device.microphone_input import (
    MicrophoneInput as TurnBasedMicrophoneInput,
)
from vocode.turn_based.output_device.speaker_output import SpeakerOutput as TurnBasedSpeakerOutput


def _get_device_prompt(device_infos: List[dict]) -> str:
    return """Please select a device:
{}
Choice: """.format(
        "\n".join(f"{index}: {device['name']}" for index, device in enumerate(device_infos))
    )


def create_streaming_microphone_input_and_speaker_output(
    use_default_devices: bool = False,
    input_device_name: Optional[str] = None,
    output_device_name: Optional[str] = None,
    mic_sampling_rate=None,
    speaker_sampling_rate=None,
):
    return _create_microphone_input_and_speaker_output(
        microphone_class=StreamingMicrophoneInput,
        speaker_class=BlockingStreamingSpeakerOutput,
        use_default_devices=use_default_devices,
        input_device_name=input_device_name,
        output_device_name=output_device_name,
        mic_sampling_rate=mic_sampling_rate,
        speaker_sampling_rate=speaker_sampling_rate,
    )


def create_turn_based_microphone_input_and_speaker_output(
    use_default_devices: bool = False,
    input_device_name: Optional[str] = None,
    output_device_name: Optional[str] = None,
    mic_sampling_rate=None,
    speaker_sampling_rate=None,
):
    return _create_microphone_input_and_speaker_output(
        microphone_class=TurnBasedMicrophoneInput,
        speaker_class=TurnBasedSpeakerOutput,
        use_default_devices=use_default_devices,
        input_device_name=input_device_name,
        output_device_name=output_device_name,
        mic_sampling_rate=mic_sampling_rate,
        speaker_sampling_rate=speaker_sampling_rate,
    )


def _create_microphone_input_and_speaker_output(
    microphone_class: typing.Type[Union[StreamingMicrophoneInput, TurnBasedMicrophoneInput]],
    speaker_class: typing.Type[
        Union[
            BlockingStreamingSpeakerOutput,
            TurnBasedSpeakerOutput,
        ]
    ],
    use_default_devices: bool,
    input_device_name: Optional[str],
    output_device_name: Optional[str],
    mic_sampling_rate=None,
    speaker_sampling_rate=None,
) -> Union[
    Tuple[
        StreamingMicrophoneInput,
        BlockingStreamingSpeakerOutput,
    ],
    Tuple[TurnBasedMicrophoneInput, TurnBasedSpeakerOutput],
]:
    device_infos = sd.query_devices()
    input_device_infos = list(
        filter(lambda device_info: device_info["max_input_channels"] > 0, device_infos)
    )
    output_device_infos = list(
        filter(lambda device_info: device_info["max_output_channels"] > 0, device_infos)
    )

    if use_default_devices:
        input_device_info = sd.query_devices(kind="input")
        output_device_info = sd.query_devices(kind="output")
    else:
        if input_device_name is not None:
            input_device_info = _find_device_with_name(input_device_infos, input_device_name)
        else:
            input_device_info = input_device_infos[
                int(input(_get_device_prompt(input_device_infos)))
            ]

        if output_device_name is not None:
            output_device_info = _find_device_with_name(output_device_infos, output_device_name)
        else:
            output_device_info = output_device_infos[
                int(input(_get_device_prompt(output_device_infos)))
            ]

    if logger is not None:
        logger.info(f"Using microphone input device: {input_device_info['name']}")

    microphone_input = microphone_class(input_device_info, sampling_rate=mic_sampling_rate)
    if logger is not None:
        logger.info(f"Using speaker output device: {output_device_info['name']}")

    speaker_output = speaker_class(output_device_info, sampling_rate=speaker_sampling_rate)
    return microphone_input, speaker_output  # type: ignore


def _find_device_with_name(device_infos: List[dict], name: str) -> dict:
    try:
        return next(filter(lambda device_info: name == device_info["name"], device_infos))
    except StopIteration:
        raise Exception("Could not find device with name: {}".format(name))
