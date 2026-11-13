from estimate_budget import estimate, main


def test_estimate_uses_query_config_product():
    result = estimate(10, 4, 400, 600, 2500, 20_000, 1, 1.8)

    assert result["instances"] == 40
    assert result["gpu_hours"] > 0
    assert result["usd"] == result["gpu_hours"] * 1.8


def test_cli_accepts_documented_flags(capsys):
    main(["--n-queries", "2", "--n-configs", "3"])

    output = capsys.readouterr().out
    assert "Instancias: 6" in output
    assert "GPU-horas" in output


def test_cli_keeps_legacy_n_alias(capsys):
    main(["--n", "2", "--n_cfgs", "3"])

    assert "Instancias: 6" in capsys.readouterr().out
