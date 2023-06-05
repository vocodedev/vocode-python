import os
from jinja2 import Environment, FileSystemLoader
from fastapi import Response


class Templater:
    def __init__(self):
        self.templates = Environment(
            loader=FileSystemLoader("%s/templates/" % os.path.dirname(__file__))
        )

    def render_template(self, template_name: str, **kwargs):
        template = self.templates.get_template(template_name)
        return template.render(**kwargs)

    def get_connection_twiml(self, call_id: str, base_url: str):
        return Response(
            self.render_template("connect_call.xml", base_url=base_url, id=call_id),
            media_type="application/xml",
        )

    def get_play_digits_and_connect_call_twiml(self, call_id: str, base_url: str, digits: str):
        # Note that this returns just the raw TWIML, not a Response object.
        return self.render_template(
            "play_digits_and_connect_call.xml",
            base_url=base_url,
            id=call_id,
            digits=digits,
        )

    def get_play_digits_twiml(self, digits: str):
        # Note that this returns just the raw TWIML, not a Response object.
        return self.render_template(
            "play_digits.xml",
            digits=digits,
        )