def test_logger():
    import tempfile
    from tblogging import TBLogger
    with tempfile.TemporaryDirectory() as logdir:
        logger = TBLogger(logdir, "test")
        logger.register_scalar("mean", "scalar")
        logger.freeze()
        logger.log(1, {"mean": 0.0})
        logger.close()
