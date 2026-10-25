"""Experimental timeline of the play, wait, section, caption, and sound calls of a scene."""

from __future__ import annotations

import contextlib
import hashlib
import inspect
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

__all__ = ["Timeline"]

_SCHEMA = "manim.timeline"
_VERSION = 1
_PACKAGE = Path(__file__).resolve().parent


def _dumps(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)


@dataclass(frozen=True, slots=True)
class Timeline:
    """A timeline recorded while a scene was evaluated or rendered.

    Get a timeline from :attr:`.Manager.timeline` after calling
    :meth:`.Manager.evaluate` or :meth:`.Manager.render` with
    ``capture_timeline=True``, or load one with :meth:`from_json`. A timeline
    cannot be modified; use :meth:`to_dict` to get its data as a dictionary you
    can edit.

    The format and Python API are experimental and may change without a
    deprecation period. See :doc:`/guides/timeline` for a description of the
    format.

    Parameters
    ----------
    _document
        The timeline as JSON text. Use :meth:`from_json` to load a timeline.
    """

    _document: str

    def __post_init__(self) -> None:
        data = json.loads(self._document)
        if (
            not isinstance(data, dict)
            or data.get("schema") != _SCHEMA
            or data.get("version") != _VERSION
        ):
            raise ValueError(f"Expected a {_SCHEMA} document with version {_VERSION}.")

    @classmethod
    def from_json(cls, document: str) -> Timeline:
        """Load a timeline from JSON text.

        Parameters
        ----------
        document
            The JSON text of a timeline, for example as written by :meth:`write`.

        Raises
        ------
        ValueError
            If the document is not a version 1 timeline.
        """
        return cls(_dumps(json.loads(document)))

    def to_dict(self) -> dict[str, Any]:
        """Return the timeline data as a new dictionary."""
        return cast("dict[str, Any]", json.loads(self._document))

    def to_json(self) -> str:
        """Return the timeline as JSON text with a trailing newline.

        The same timeline always gives the same text.
        """
        return self._document + "\n"

    def write(self, path: str | Path) -> None:
        """Save the timeline as a UTF-8 JSON file.

        An existing file is replaced only after the new file has been written
        completely.

        Parameters
        ----------
        path
            The destination file. Missing parent directories are created.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
            ) as stream:
                temporary = Path(stream.name)
                stream.write(self.to_json().encode("utf-8"))
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
        finally:
            if temporary is not None:
                with contextlib.suppress(OSError):
                    temporary.unlink()


def _primary_source(scene_class: type) -> tuple[dict[str, Any] | None, Path]:
    """Return the ``source`` entry for a scene class and the source root."""
    try:
        filename = inspect.getsourcefile(scene_class)
    except (OSError, TypeError):
        filename = None
    if filename is not None:
        path = Path(filename).resolve()
        with contextlib.suppress(OSError):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            return {"path": path.name, "sha256": digest}, path.parent
    return None, Path.cwd().resolve()


class _TimelineRecorder:
    """Collect timeline entries while a Manager runs a scene."""

    def __init__(
        self, scene: Any, mode: Literal["evaluate", "render"], frame_rate: float
    ) -> None:
        self.mode = mode
        self.scene_name = type(scene).__qualname__
        self.backend = type(scene.renderer).__name__
        self.frame_rate = float(frame_rate)
        self.source, self.root = _primary_source(type(scene))
        self.entries: list[dict[str, Any]] = []
        self.active: dict[str, Any] | None = None
        self.preparing = False
        self.last_time = 0.0
        self.failed = False
        self.ended_early = False

    def location(self) -> dict[str, Any] | None:
        """Return the file and line of the scene code that made the current call."""
        frame = inspect.currentframe()
        try:
            while frame is not None:
                filename = frame.f_code.co_filename
                if filename.startswith("<"):
                    return None
                path = Path(filename).resolve()
                if not path.is_relative_to(_PACKAGE):
                    relative = (
                        path.relative_to(self.root).as_posix()
                        if path.is_relative_to(self.root)
                        else None
                    )
                    return {"path": relative, "line": frame.f_lineno}
                frame = frame.f_back
            return None
        finally:
            # Break the reference cycle between this frame and its callers.
            del frame

    def observe_time(self, time: float) -> None:
        if not math.isfinite(time) or time < self.last_time:
            self.failed = True
            raise ValueError(
                "Timeline capture requires finite scene time that never goes backwards."
            )
        self.last_time = float(time)

    def enter(self, time: float) -> None:
        """Start recording a play or wait call, before its animations are prepared."""
        self.observe_time(time)
        if self.active is not None or self.preparing:
            self.failed = True
            raise RuntimeError(
                "Timeline capture does not support play or wait calls made "
                "during another play or wait call."
            )
        self.preparing = True

    def begin(self, scene: Any, start: float) -> None:
        """Add the entry for a play or wait call once its animations are prepared."""
        from .animation.animation import Wait

        animations = scene.animations or []
        kind = (
            "wait"
            if len(animations) == 1 and isinstance(animations[0], Wait)
            else "play"
        )
        summaries = []
        for animation in animations:
            cls = type(animation)
            name = (
                f"{cls.__module__}.{cls.__qualname__}"
                if cls.__module__.startswith("manim.")
                else cls.__qualname__
            )
            summaries.append({"type": name, "run_time": float(animation.run_time)})
        entry = {
            "kind": kind,
            "start": float(start),
            "end": None,
            "run_time": float(scene.duration),
            "frozen": scene.is_current_animation_frozen_frame(),
            "animations": summaries,
            "skipped": False,
            "source": self.location(),
        }
        self.entries.append(entry)
        self.active = entry
        self.preparing = False

    def end(self, time: float, *, skipped: bool) -> None:
        """Finish the entry of the current play or wait call."""
        assert self.active is not None
        self.observe_time(time)
        self.active["end"] = float(time)
        self.active["skipped"] = skipped
        self.active = None

    def add(self, kind: str, time: float, **values: Any) -> None:
        """Add a section, caption, or sound entry."""
        self.observe_time(time)
        try:
            # Copy the values, so that later changes by scene code are not recorded.
            values = json.loads(json.dumps(values, allow_nan=False))
        except (TypeError, ValueError):
            self.failed = True
            raise
        self.entries.append({"kind": kind, **values, "source": self.location()})

    def finish(self, time: float) -> Timeline:
        if self.failed:
            raise RuntimeError("Timeline capture failed earlier during this run.")
        if self.active is not None or self.preparing:
            raise RuntimeError("Timeline capture ended during a play or wait call.")
        self.observe_time(time)
        data = {
            "schema": _SCHEMA,
            "version": _VERSION,
            "mode": self.mode,
            "scene": self.scene_name,
            "backend": self.backend,
            "frame_rate": self.frame_rate,
            "duration": float(time),
            "ended_early": self.ended_early,
            "source": self.source,
            "entries": self.entries,
        }
        return Timeline(_dumps(data))
