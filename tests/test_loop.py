import signal

from nanodiffusion.signals import install_stop_handlers


def test_stop_handler_records_sigterm_and_restores_previous_handler() -> None:
    previous = signal.getsignal(signal.SIGTERM)
    with install_stop_handlers() as stop:
        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler)
        typed_handler = handler
        typed_handler(signal.SIGTERM, None)
        assert stop.signum == signal.SIGTERM
        assert stop.requested

    assert signal.getsignal(signal.SIGTERM) == previous
