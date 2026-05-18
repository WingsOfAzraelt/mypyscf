#!/usr/bin/env python

from __future__ import annotations

import unittest
from unittest import mock

from pyscf_agent import pyscf_agent_backend as backend


class DummyWorkflow:
    '''Minimal workflow stub used to verify LangGraph orchestration calls.'''

    def __init__(self):
        self.calls = []

    def invoke(self, state):
        self.calls.append(state)
        final_state = dict(state)
        final_state['execution_status'] = 'stubbed'
        final_state['final_report'] = {
            'channel': state.get('channel'),
            'messages': list(state.get('messages', [])),
        }
        return final_state


class BackendWorkflowTests(unittest.TestCase):
    def setUp(self):
        backend.get_workflow.cache_clear()

    def tearDown(self):
        backend.get_workflow.cache_clear()

    def test_run_workflow_invokes_supplied_workflow(self):
        workflow = DummyWorkflow()
        initial_state = backend.default_state('atom: H 0 0 0', channel='test')

        result = backend.run_workflow(initial_state, workflow=workflow)

        self.assertEqual(result['execution_status'], 'stubbed')
        self.assertEqual(len(workflow.calls), 1)
        self.assertEqual(workflow.calls[0]['channel'], 'test')

    def test_execute_request_uses_cached_langgraph_workflow_by_default(self):
        workflow = DummyWorkflow()
        with mock.patch.object(backend, 'build_workflow', return_value=workflow) as mock_build_workflow:
            first = backend.execute_request('basis: sto-3g', channel='cli')
            second = backend.execute_request('basis: 6-31g', channel='web')

        self.assertEqual(mock_build_workflow.call_count, 1)
        self.assertEqual(first['execution_status'], 'stubbed')
        self.assertEqual(second['execution_status'], 'stubbed')
        self.assertEqual(workflow.calls[0]['channel'], 'cli')
        self.assertEqual(workflow.calls[1]['channel'], 'web')

    def test_run_workflow_sequential_keeps_existing_pipeline_behavior(self):
        result = backend.run_workflow_sequential(backend.default_state('basis: sto-3g', channel='test'))

        self.assertEqual(result['execution_status'], 'blocked')
        self.assertIn('Missing molecular geometry', result['raw_stderr'])
        self.assertIn('final_report', result)


if __name__ == '__main__':
    unittest.main()
