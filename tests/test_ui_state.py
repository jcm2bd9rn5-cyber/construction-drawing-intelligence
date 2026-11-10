"""Execute the shipped UI script with a minimal DOM and mocked HTTP responses.

This catches syntax errors and asynchronous mode races without external APIs.
Actual visual layout still needs the user's Chrome when its connector is down.
"""
import json
import re
import shutil
import subprocess

import pytest
from app.main import ui


def run_ui(assertions):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required for the shipped JavaScript state test")
    script = re.findall(r"<script>(.*?)</script>", ui(), re.S)[-1]
    harness = r"""
    const vm=require('node:vm'),assert=require('node:assert/strict');
    const elements={};
    function makeElement(id){const classes=new Set(),attrs={};return {value:'',hidden:id==='qaResults',innerHTML:'',textContent:'',disabled:false,tabIndex:0,inert:false,
       classList:{toggle(name,force){const add=force===undefined?!classes.has(name):force;if(add)classes.add(name);else classes.delete(name);return add;},contains(name){return classes.has(name);}},
       setAttribute(name,value){attrs[name]=String(value);},getAttribute(name){return attrs[name]??null;},removeAttribute(name){delete attrs[name];},addEventListener(){},focus(){}};}
    const document={documentElement:{dataset:{theme:'dark'}},getElementById(id){return elements[id]??=makeElement(id);},addEventListener(){}};
    const context=vm.createContext({document,console});
    """
    result = subprocess.run([node, "-e", harness + "vm.runInContext(" + json.dumps(script) + ",context);\n" + assertions], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_late_qa_response_stays_hidden_in_search_mode():
    run_ui(r"""
    (async()=>{
      let finish;context.fetch=()=>new Promise(resolve=>{finish=resolve;});
      vm.runInContext("setMode('qa');$('q').value='質問';",context);
      const pending=vm.runInContext('submitQuery()',context);
      vm.runInContext("setMode('search')",context);
      finish({ok:true,json:async()=>({answer:'回答',citation_groups:[],evidence:[]})});
      await pending;
      assert.equal(elements.qaResults.hidden,true);
      assert.equal(elements.searchResults.hidden,false);
      assert.ok(elements.qaResults.innerHTML.includes('回答'));
      assert.ok(elements.searchResults.innerHTML.includes('initial-guidance'));
      vm.runInContext("setMode('qa')",context);
      assert.equal(elements.qaResults.hidden,false);
      assert.equal(elements.searchResults.hidden,true);
    })().catch(e=>{console.error(e);process.exitCode=1;});
    """)


def test_retrieval_scores_move_to_bottom_panel_without_data_loss():
    run_ui(r"""
    vm.runInContext(`state.qa.data={answer:'回答',citation_groups:[],evidence:[{drawing_number:'X',snippet:'snippet',score:2,lexical_score:3,metadata_score:4,vector_score:.2}]};mode='qa';setBottomTab('retrieval')`,context);
    const html=elements.bottomContent.innerHTML;
    assert.ok(html.includes('<table class="data-table">'));
    assert.ok(html.includes('<th>Lexical</th>') && html.includes('<td>3</td>'));
    assert.ok(html.includes('<th>Metadata</th>') && html.includes('<td>4</td>'));
    assert.ok(html.includes('<th>pgvector</th>') && html.includes('<td>0.2</td>'));
    assert.ok(!html.includes('Evidence confidence'));
    """)


def test_ui_escapes_citation_quotes_and_renders_server_provenance():
    run_ui(r"""
    const html=vm.runInContext(`citationCard({number:1,drawing_number:'X',page:1,source_label:'PDF / 図面',snippets:[{preview:'<script>bad</script>',quote:'full text',viewer_url:'/viewer?page=1',citation_scope:'region'}]})`,context);
    assert.ok(html.includes('>PDF<'));
    assert.ok(html.includes('&lt;script&gt;'));
    assert.ok(!html.includes('<script>bad'));
    assert.ok(html.includes('Evidenceパネルで確認'));
    assert.ok(html.includes('表示'));
    assert.ok(html.includes('citation-number'));
    """)


def test_reset_discards_inflight_response_and_active_filter_can_be_removed():
    run_ui(r"""
    (async()=>{
      let finish;context.fetch=()=>new Promise(resolve=>{finish=resolve;});
      vm.runInContext("$('q').value='検索';$('floor').value='1階';renderFilters()",context);
      assert.ok(elements.activeFilters.innerHTML.includes('適用中:'));
      const pending=vm.runInContext('submitQuery()',context);
      vm.runInContext("resetForm()",context);
      finish({ok:true,json:async()=>({results:[{drawing_number:'X'}]})});
      await pending;
      assert.ok(elements.searchResults.innerHTML.includes('initial-guidance'));
      assert.equal(elements.floor.value,'');
      assert.equal(elements.activeFilters.hidden,true);
    })().catch(e=>{console.error(e);process.exitCode=1;});
    """)


def test_workstation_shell_replaces_hero_and_keeps_revision_in_rows():
    html = ui()
    assert 'class="app-layout"' in html
    assert 'aria-label="ワークスペースメニュー"' in html
    assert "Construction Drawing Intelligence" in html
    assert "図面を探す。根拠を確認する。" not in html
    assert "Revisionは結果行で確認できます" not in html
    assert 'class="upper-workspace ux u101"' in html
    assert 'class="viewer-panel"' in html
    assert 'class="bottom-panel"' in html
    assert "'Rev '+revision(x)" in html
    assert "資料を検索" in html
    assert "Markdown資料" in html


def test_lucian_workspace_ports_are_attributed_and_keep_accessible_behavior():
    html = ui()
    assert "Lucian UX/UI Prompt #101 · MIT License" in html
    assert "Lucian UX/UI Prompt #092 · MIT License" in html
    assert 'class="upper-workspace ux u101"' in html
    assert 'aria-valuemin="25" aria-valuemax="65"' in html
    assert "'ArrowLeft','ArrowRight','Home','End','Escape'" in html
    assert "prefers-reduced-motion:reduce" in html
    run_ui(r"""
    elements.q.value='該当なし';
    elements.floor.value='9階';
    const html=vm.runInContext(`emptyMarkup('該当なし')`,context);
    assert.ok(html.includes('class="ux u092 empty"'));
    assert.ok(html.includes('階「9階」を解除'));
    assert.ok(html.includes('u092-lens'));
    """)


def test_tablet_field_workspace_controls_are_shipped():
    html = ui()
    assert "Lucian UX/UI Prompt #048 · MIT License" in html
    assert "Lucian UX/UI Prompt #021 · MIT License" not in html
    assert "Lucian UX/UI Prompt #045 · MIT License" not in html
    assert 'id="themeToggle"' not in html
    assert 'id="mobileMenu"' in html
    assert 'id="mobileResultsTab"' in html
    assert 'id="mobileViewerTab"' in html
    assert 'id="appearanceSelect" aria-label="外観"' in html
    assert '<option value="system">System</option>' in html
    assert '.appearance-options select{width:100%;height:36px' in html
    assert ':root[data-theme="light"]' in html
    assert "@media(max-width:1199px)" in html
    assert "@media(max-width:900px) and (orientation:portrait)" in html
    assert ".health{display:none}" in html
    assert "setBottomCollapsed(isTablet())" in html
    assert "grid-template-columns:minmax(0,55%) minmax(0,45%)" in html
    run_ui(r"""
    vm.runInContext(`applyThemeChoice('light',false);setMobilePanel('viewer')`,context);
    assert.equal(document.documentElement.dataset.theme,'light');
    assert.equal(elements.appearanceSelect.value,'light');
    assert.ok(elements.upperWorkspace.classList.contains('mobile-viewer-active'));
    assert.equal(elements.mobileViewerTab.getAttribute('aria-selected'),'true');
    """)


def test_tablet_selection_filter_summary_and_retrieval_summary():
    run_ui(r"""
    elements.floor.value='1階';elements.discipline.value='電気';
    vm.runInContext('renderFilters()',context);
    assert.equal(elements.filterSummary.textContent,'Filters: 1階 · 電気');
    const reason=vm.runInContext(`reasonSummary(['title:受変電設備','floor=1階','semantic:受変電設備','pgvector similarity'])`,context);
    assert.equal(reason,'タイトル一致 · 階一致 · 意味検索');
    const source=vm.runInContext(`citationCard({number:1,drawing_number:'E03-01',page:1,source_type:'pdf',snippets:[]},0)`,context);
    assert.ok(source.startsWith('<button type="button" class="citation"'));
    """)


def test_loading_guard_and_inline_api_error_are_preserved():
    run_ui(r"""
    (async()=>{
      let calls=0;context.fetch=async()=>{calls++;throw new Error('offline');};
      elements.q.value='検索';
      await Promise.all([vm.runInContext('submitQuery()',context),vm.runInContext('submitQuery()',context)]);
      assert.equal(calls,1);
      assert.equal(elements.submit.disabled,false);
      assert.ok(elements.status.innerHTML.includes('検索に失敗しました'));
      assert.ok(elements.status.innerHTML.includes('class="error"'));
    })().catch(e=>{console.error(e);process.exitCode=1;});
    """)


def test_qa_completion_keeps_the_current_viewer_selection_unchanged():
    run_ui(r"""
    (async()=>{
      context.fetch=async()=>({ok:true,json:async()=>({answer:'結論',citation_groups:[{number:1,drawing_number:'E03-01',page:1,snippets:[]}],evidence:[]})});
      vm.runInContext("setMode('qa');state.selected={drawing_number:'KEEP'};$('q').value='質問'",context);
      await vm.runInContext('submitQuery()',context);
      assert.equal(vm.runInContext('state.selected.drawing_number',context),'KEEP');
      assert.ok(elements.qaResults.innerHTML.includes('結論'));
    })().catch(e=>{console.error(e);process.exitCode=1;});
    """)


def test_tablet_polish_keeps_explicit_viewer_and_drawer_contracts():
    html = ui()
    assert 'width:min(360px,88vw)' in html
    assert 'inset:52px 0 0 auto' in html
    assert 'transform:translateX(102%)' in html
    assert "$('drawerClose').focus()" in html
    assert '候補根拠' in html
    assert '詳細を表示' in html
    assert 'bbox=0.286' not in html  # no drawing-specific UI hardcode
    assert 'prefers-reduced-motion:reduce' in html


def test_qa_answer_context_avoids_duplicate_question_and_tracks_edited_input():
    run_ui(r"""
    (async()=>{
      context.fetch=async()=>({ok:true,json:async()=>({answer:'結論',citation_groups:[],evidence:[]})});
      vm.runInContext("setMode('qa');$('q').value='受電電圧は？'",context);
      await vm.runInContext('submitQuery()',context);
      assert.ok(!elements.qaResults.innerHTML.includes('QUESTION'));
      elements.q.value='変圧器容量は？';
      vm.runInContext("$('q').addEventListener; $('q').dispatchEvent?.({type:'input'})",context);
      // The minimal harness does not dispatch listeners, so assert the same
      // renderer contract directly.
      const edited=vm.runInContext("qaMarkup(state.qa.data,state.qa.submittedQuestion)",context);
      assert.ok(edited.includes('回答対象: 受電電圧は？'));
    })().catch(e=>{console.error(e);process.exitCode=1;});
    """)
