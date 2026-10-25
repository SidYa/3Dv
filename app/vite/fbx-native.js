
// --- 3Dv: appended to three's FBXLoader.js by vite.config.ts ------------------
//
// The viewer's FBX fast path. Reading a binary FBX and building its mesh
// geometry - nearly all of FBXLoader's time - happens in Rust
// (core/render/src/viewer_fbx.rs), which sends the file's node list and the
// finished vertex buffers. Here BinaryParser's own code turns the node list
// into FBXLoader's tree and the unchanged FBXTreeParser builds the scene from
// it: materials, textures, skeletons and animations come out exactly as
// FBXLoader makes them. Only GeometryParser's buffers are taken from Rust.
//
// This code sees the module's internals (fbxTree, BinaryParser,
// GeometryParser...), which is why it is appended rather than imported.

/**
 * FBXLoader.parse() for a "3DVF" blob from the load_model command.
 *
 * @param {FBXLoader} loader - Supplies the manager and options, as for parse().
 * @param {ArrayBuffer} buffer - The blob.
 * @param {string} path - The URL base textures resolve against.
 * @return {Group}
 */
export function parseNativeFBX( loader, buffer, path ) {

	const view = new DataView( buffer );
	if ( view.getUint32( 0, true ) !== 0x46564433 ) throw new Error( 'THREE.FBXLoader: not a native FBX blob' );
	const jsonLength = view.getUint32( 4, true );
	const header = JSON.parse( new TextDecoder().decode( new Uint8Array( buffer, 8, jsonLength ) ) );
	const base = ( 8 + jsonLength + 7 ) & ~ 7;

	// Property values as BinaryParser.parseProperty() returns them.
	const value = ( v ) => {

		if ( ! Array.isArray( v ) ) return v;
		switch ( v[ 0 ] ) {

			case 'n': return Number( v[ 1 ] );
			case 'r': return buffer.slice( base + v[ 1 ], base + v[ 1 ] + v[ 2 ] );
			case 'f': return Array.from( new Float64Array( buffer, base + v[ 1 ], v[ 2 ] ) );
			case 'b': return Array.from( new Uint8Array( buffer, base + v[ 1 ], v[ 2 ] ), ( x ) => x === 1 );
			case 'x': return []; // geometry data, built in Rust

		}

		throw new Error( 'THREE.FBXLoader: unknown native property' );

	};

	// BinaryParser.parseNode() over an already read node.
	const binaryParser = new BinaryParser();
	const parseNode = ( [ name, props, children, singleProperty ] ) => {

		const node = {};
		const propertyList = props.map( value );
		const id = propertyList.length > 0 ? propertyList[ 0 ] : '';
		const attrName = propertyList.length > 1 ? propertyList[ 1 ] : '';
		const attrType = propertyList.length > 2 ? propertyList[ 2 ] : '';
		node.singleProperty = singleProperty;
		for ( const child of children ) binaryParser.parseSubNode( name, node, parseNode( child ) );
		node.propertyList = propertyList;
		if ( typeof id === 'number' ) node.id = id;
		if ( attrName !== '' ) node.attrName = attrName;
		if ( attrType !== '' ) node.attrType = attrType;
		if ( name !== '' ) node.name = name;
		return node;

	};

	fbxTree = new FBXTree();
	for ( const raw of header.nodes ) {

		const node = parseNode( raw );
		fbxTree.add( node.name, node );

	}

	const floats = ( [ offset, count ] ) => new Float32Array( buffer, base + offset, count );
	const geometries = new Map( header.geometries.map( ( g ) => [ g.id, g ] ) );

	// GeometryParser with genGeometry() / genMorphGeometry() taking the
	// buffers from Rust; anything Rust did not build goes the usual way.
	class NativeGeometryParser extends GeometryParser {

		genGeometry( geoNode, skeleton, morphTargets, preTransform ) {

			const native = geometries.get( geoNode.id );
			if ( native === undefined ) return super.genGeometry( geoNode, skeleton, morphTargets, preTransform );
			if ( String( native.skin ) !== String( skeleton ? skeleton.ID : null ) ) throw new Error( 'THREE.FBXLoader: native skin mismatch' );

			const geo = new BufferGeometry();
			if ( geoNode.attrName ) geo.name = geoNode.attrName;

			const positionAttribute = new Float32BufferAttribute( floats( native.position ), 3 );
			positionAttribute.applyMatrix4( preTransform );
			geo.setAttribute( 'position', positionAttribute );

			if ( native.color ) geo.setAttribute( 'color', new Float32BufferAttribute( floats( native.color ), 3 ) );

			if ( skeleton ) {

				const [ offset, count ] = native.skinIndex;
				geo.setAttribute( 'skinIndex', new Uint16BufferAttribute( new Uint16Array( buffer, base + offset, count ), 4 ) );
				geo.setAttribute( 'skinWeight', new Float32BufferAttribute( floats( native.skinWeight ), 4 ) );
				geo.FBX_Deformer = skeleton;

			}

			if ( native.normal ) {

				const normalMatrix = new Matrix3().getNormalMatrix( preTransform );
				const normalAttribute = new Float32BufferAttribute( floats( native.normal ), 3 );
				normalAttribute.applyNormalMatrix( normalMatrix );
				geo.setAttribute( 'normal', normalAttribute );

			}

			native.uvs.forEach( ( uv, i ) => geo.setAttribute( i === 0 ? 'uv' : `uv${ i }`, new Float32BufferAttribute( floats( uv ), 2 ) ) );

			if ( native.groups ) {

				for ( const [ start, count, materialIndex ] of native.groups ) geo.addGroup( start, count, materialIndex );

			}

			this.addMorphTargets( geo, geoNode, morphTargets, preTransform );
			return geo;

		}

		genMorphGeometry( parentGeo, parentGeoNode, morphGeoNode, preTransform, name ) {

			const native = geometries.get( parentGeoNode.id );
			const morph = native && native.morphs.find( ( m ) => m[ 0 ] === morphGeoNode.id );
			if ( ! morph ) throw new Error( 'THREE.FBXLoader: native morph target missing' );

			const positionAttribute = new Float32BufferAttribute( floats( morph[ 1 ] ), 3 );
			positionAttribute.name = name || morphGeoNode.attrName;
			positionAttribute.applyMatrix4( preTransform );
			parentGeo.morphAttributes.position.push( positionAttribute );

		}

	}

	// FBXLoader.parse() and FBXTreeParser.parse() from here on.
	const textureLoader = new TextureLoader( loader.manager ).setPath( loader.resourcePath || path ).setCrossOrigin( loader.crossOrigin );
	const treeParser = new FBXTreeParser( textureLoader, loader.manager, loader.trimAnimationClips );

	connections = treeParser.parseConnections();
	const images = treeParser.parseImages();
	const textures = treeParser.parseTextures( images );
	const materials = treeParser.parseMaterials( textures );
	const deformers = treeParser.parseDeformers();
	const geometryMap = new NativeGeometryParser().parse( deformers );
	treeParser.parseScene( deformers, geometryMap, materials );

	return sceneGraph;

}
