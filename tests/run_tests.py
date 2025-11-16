#!/usr/bin/env python
"""
Script para executar todos os testes unitários da SmartMonitor API.

Este script executa todos os testes usando unittest, fornece estatísticas
detalhadas e formatação colorida de resultados.

Uso:
    python tests/run_tests.py
    python tests/run_tests.py -v        # Modo verbose
    python tests/run_tests.py --quiet   # Modo silencioso
"""

import sys
import unittest
import time
from io import StringIO


def run_all_tests(verbosity=2):
    """
    Executa todos os testes unitários do projeto.
    
    Args:
        verbosity (int): Nível de verbosidade (0=quiet, 1=normal, 2=verbose)
    
    Returns:
        unittest.TestResult: Resultado dos testes
    """
    print("=" * 80)
    print("🧪 EXECUTANDO TESTES UNITÁRIOS - SmartMonitor API")
    print("=" * 80)
    print()
    
    # Descobrir todos os testes no diretório tests/
    loader = unittest.TestLoader()
    start_dir = 'tests'
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Contar testes
    total_tests = suite.countTestCases()
    print(f"📊 Total de testes encontrados: {total_tests}")
    print()
    
    # Executar testes
    print("🚀 Iniciando execução dos testes...")
    print("-" * 80)
    print()
    
    start_time = time.time()
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    end_time = time.time()
    
    # Estatísticas
    print()
    print("=" * 80)
    print("📈 ESTATÍSTICAS DOS TESTES")
    print("=" * 80)
    print(f"✅ Testes executados: {result.testsRun}")
    print(f"✔️  Sucessos: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ Falhas: {len(result.failures)}")
    print(f"💥 Erros: {len(result.errors)}")
    print(f"⏱️  Tempo total: {end_time - start_time:.2f}s")
    print()
    
    # Detalhes de falhas
    if result.failures:
        print("=" * 80)
        print("❌ FALHAS DETALHADAS")
        print("=" * 80)
        for test, traceback in result.failures:
            print(f"\n🔴 {test}:")
            print(traceback)
    
    # Detalhes de erros
    if result.errors:
        print("=" * 80)
        print("💥 ERROS DETALHADOS")
        print("=" * 80)
        for test, traceback in result.errors:
            print(f"\n🔴 {test}:")
            print(traceback)
    
    # Resultado final
    print("=" * 80)
    if result.wasSuccessful():
        print("🎉 RESULTADO: TODOS OS TESTES PASSARAM!")
        print("=" * 80)
        return 0
    else:
        print("⚠️  RESULTADO: ALGUNS TESTES FALHARAM")
        print("=" * 80)
        return 1


def print_test_files():
    """Imprime lista de arquivos de teste disponíveis."""
    import os
    
    print("📁 Arquivos de teste disponíveis:")
    print("-" * 80)
    
    tests_dir = 'tests'
    test_files = [f for f in os.listdir(tests_dir) if f.startswith('test_') and f.endswith('.py')]
    
    for i, test_file in enumerate(test_files, 1):
        print(f"{i}. {test_file}")
    
    print()


def main():
    """Função principal."""
    # Parsear argumentos
    verbosity = 2  # verbose por padrão
    
    if '--help' in sys.argv or '-h' in sys.argv:
        print(__doc__)
        print_test_files()
        return 0
    
    if '--quiet' in sys.argv or '-q' in sys.argv:
        verbosity = 0
    elif '-v' in sys.argv:
        verbosity = 2
    
    # Executar testes
    exit_code = run_all_tests(verbosity)
    
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
